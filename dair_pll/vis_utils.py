"""Utility functions for visualizing trajectories.

Visualization of Drake systems can be done with Drake's VideoWriter. This allows
for a relatively thin implementation of visualization for very complex
geometries.

The main contents of this file are as follows:

    * A method to generate a dummy ``DrakeSystem`` which simultaneously
      visualizes two trajectories of the same system.
    * A method which takes a ``DrakeSystem`` and corresponding trajectory,
      captures a visualization video, and outputs it as a numpy ndarray.
"""
from copy import deepcopy
from typing import Tuple, Optional

import numpy as np
import torch
from torch import Tensor
from PIL import Image
# pylint: disable-next=import-error
from pydrake.geometry import Role, RoleAssign, Rgba  # type: ignore
from torch import Tensor
from tempfile import TemporaryDirectory

from dair_pll.drake_system import DrakeSystem
from dair_pll import file_utils

RESOLUTION = [640, 480]
RED = Rgba(0.6, 0.0, 0.0, 0.5)
BLUE = Rgba(0.0, 0.0, 0.6, 0.7)
BASE_SYSTEM_DEFAULT_COLOR = RED
LEARNED_SYSTEM_DEFAULT_COLOR = BLUE
PERCEPTION_COLOR_GROUP = 'phong'
PERCEPTION_COLOR_PROPERTY = 'diffuse'
LEARNED_TAG = '__learned__'
GEOMETRY_INSPECTION_TRAJECTORY_LENGTH = 180
HALF_STEPS = int(GEOMETRY_INSPECTION_TRAJECTORY_LENGTH/2)

FULL_SPIN_HALF_TIME = torch.stack([
        Tensor([np.cos(torch.pi*i/HALF_STEPS), 0, 0,
                np.sin(torch.pi*i/HALF_STEPS)]) \
        for i in range(HALF_STEPS)
    ])
ARTICULATION_FULL_SPIN_HALF_TIME = torch.stack([
        Tensor([2*torch.pi*i/HALF_STEPS])
        for i in range(HALF_STEPS)
    ])
LINEAR_LOCATION_HALF_TIME = Tensor([1.2, 0, 0.15]).repeat(HALF_STEPS, 1)
ROBOT_JOINTS_HALF_TIME = Tensor([0, -1, 0, -2.5, 0, 1.5, 0]
                                ).repeat(HALF_STEPS, 1)

def add_text_to_video(video_array, text, position=(50, 50)):
    """
    Add text to video frames stored in numpy array of shape (1, t, c, h, w)
    
    Args:
        video_array: numpy array of shape (1, t, c, h, w) with dtype uint8
        text: string to write on frames
        position: tuple of (x, y) coordinates for text position
    
    Returns:
        numpy array with same shape as input, with text overlaid
    """
    # Remove batch dimension and reorder to (t, h, w, c) for OpenCV
    video = np.squeeze(video_array, axis=0)
    video = np.transpose(video, (0, 2, 3, 1))
    video = np.ascontiguousarray(video)
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_color = (255, 255, 255)  # White color in BGR
    thickness = 1
    
    # Add text to each frame
    for frame_idx in range(video.shape[0]):
        cv2.putText(video[frame_idx], 
                   text,
                   position, 
                   font,
                   font_scale,
                   font_color,
                   thickness)
    
    # Restore original shape (1, t, c, h, w)
    video = np.transpose(video, (0, 3, 1, 2))
    video = np.expand_dims(video, axis=0)
    
    return video

def read_video(video_rgb: str):
    if os.path.isdir(video_rgb):
        frames_rgb = os.listdir(video_rgb)
        frames_rgb.sort()
        frames_bgr = [cv2.imread(op.join(video_rgb, frame)) for frame in frames_rgb]
        frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
        print(f"Read {len(frames_rgb)} frames from {video_rgb}")
    elif os.path.isfile(video_rgb) and video_rgb.endswith('.gif'):
        gif = Image.open(video_rgb)
        frames_rgb = []
        for i in range(gif.n_frames):
            gif.seek(i)
            frame = gif.convert('RGB')
            frames_rgb.append(np.array(frame))
        print(f"Read {len(frames_rgb)} frames from {video_rgb}")
    elif os.path.isfile(video_rgb) and video_rgb.endswith('.mp4'):
        cap = cv2.VideoCapture(video_rgb)
        frames_rgb = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        print(f"Read {len(frames_rgb)} frames from {video_rgb}")
    else:
        raise ValueError("Unsupported video_rgb format")
    return frames_rgb

def get_geometry_inspection_trajectory(learned_system: DrakeSystem) -> Tensor:
    """Return a trajectory to use to inspect the learned geometry of a system.

    Notes:
        Only works for the cube and elbow experiments, or actually more
        generally for one floating body with or without one articulation joint.

    Args:
        learned_system: This system is used as an input only for determining the
          size of the system's state space.

    Returns:
        (n_steps, n_x) tensor of a trajectory.
    """
    n_q = learned_system.space.n_q
    n_v = learned_system.space.n_v

    # Velocities don't matter -- set to zero.
    vels = torch.zeros((HALF_STEPS, n_v))

    # Rotation and articulation depend on if there's articulation or not.
    if n_q == 7:
        trajectory = torch.cat(
            (FULL_SPIN_HALF_TIME, LINEAR_LOCATION_HALF_TIME, vels), dim=1)

    elif n_q == 8:
        rotation_piece = torch.cat(
            (FULL_SPIN_HALF_TIME, LINEAR_LOCATION_HALF_TIME,
             torch.zeros((HALF_STEPS, 1)), vels), dim=1)

        rotate_and_articulate = torch.cat(
            (FULL_SPIN_HALF_TIME, LINEAR_LOCATION_HALF_TIME,
             ARTICULATION_FULL_SPIN_HALF_TIME, vels), dim=1)

        trajectory = torch.cat((rotation_piece, rotate_and_articulate), dim=0)

    # Detect if this is a robot-object system.
    elif n_q == 14:
        trajectory = torch.cat(
            (ROBOT_JOINTS_HALF_TIME, FULL_SPIN_HALF_TIME,
             LINEAR_LOCATION_HALF_TIME, vels), dim=1)

    else:
        raise NotImplementedError(f'Don\'t know how to handle a system ' + \
            f'other than the cube or elbow or similar.')

    return trajectory


def generate_visualization_system(
    base_system: DrakeSystem,
    visualization_file: str,
    learned_system: Optional[DrakeSystem] = None,
    base_system_color: Rgba = BASE_SYSTEM_DEFAULT_COLOR,
    learned_system_color: Rgba = LEARNED_SYSTEM_DEFAULT_COLOR,
) -> DrakeSystem:
    """Generate a dummy ``DrakeSystem`` for visualizing comparisons between two
    trajectories of ``base_system``.

    Does so by generating a new ``DrakeSystem`` in which every model in the
    base system has a copy. Each illustration geometry element in these two
    copies is uniformly colored to be visually distinguishable.

    The copy of the base system can optionally be rendered in its learned
    geometry.

    Args:
        base_system: System to be visualized.
        visualization_file: Output GIF filename for trajectory video.
        learned_system: Optionally, the learned system so the predicted
          trajectory is rendered with the learned geometry.
        base_system_color: Color to repaint every thing in base system.
        learned_system_color: Color to repaint every thing in duplicated system.

    Returns:
        New ``DrakeSystem`` with doubled state and repainted elements.
    """
    # pylint: disable=too-many-locals
    # Start with true base system.
    double_urdfs = deepcopy(base_system.urdfs)

    # Either copy the base system's geometry or optionally use the learned
    # geometry.
    system_to_add = learned_system if learned_system else base_system
    double_urdfs.update({
        (k + LEARNED_TAG): v for k, v in system_to_add.urdfs.items()
    })

    visualization_system = DrakeSystem(double_urdfs,
                                       base_system.dt,
                                       visualization_file=visualization_file)

    # Recolors every perception geometry to default colors
    plant_diagram = visualization_system.plant_diagram
    plant = plant_diagram.plant
    scene_graph = plant_diagram.scene_graph
    scene_graph_context = scene_graph.GetMyContextFromRoot(
        plant_diagram.sim.get_mutable_context())
    inspector = scene_graph.model_inspector()
    for model_id in plant_diagram.model_ids:
        model_name = plant.GetModelInstanceName(model_id)
        for body_index in plant.GetBodyIndices(model_id):
            body_frame = plant.GetBodyFrameIdOrThrow(body_index)
            for geometry_id in inspector.GetGeometries(body_frame,
                                                       Role.kPerception):
                props = inspector.GetPerceptionProperties(geometry_id)
                # phong.diffuse is the name of property controlling perception
                # color.
                if props and \
                   props.HasProperty(PERCEPTION_COLOR_GROUP, \
                                     PERCEPTION_COLOR_PROPERTY):
                    # Sets color in properties.
                    props.UpdateProperty(
                        PERCEPTION_COLOR_GROUP, PERCEPTION_COLOR_PROPERTY,
                        learned_system_color
                        if LEARNED_TAG in model_name else base_system_color)
                    # Tells ``scene_graph`` to update the color.
                    plant_source_id = plant.get_source_id()
                    assert plant_source_id is not None

                    scene_graph.RemoveRole(scene_graph_context, plant_source_id,
                                           geometry_id, Role.kPerception)
                    scene_graph.AssignRole(scene_graph_context, plant_source_id,
                                           geometry_id, props, RoleAssign.kNew)

    # Changing perception properties requires the ``Simulator`` to be
    # re-initialized.
    plant_diagram.sim.Initialize()

    return visualization_system


def visualize_trajectory(drake_system: DrakeSystem,
                         x_trajectory: Tensor,
                         framerate: int = 30, 
                         filename_postfix: str = '') -> Tuple[np.ndarray, int]:
    r"""Visualizes trajectory of system.

    Specifies a ``framerate`` for output video, though should be noted that
    this framerate is only approximately represented by homogeneous integer
    downsampling of the state trajectory. For example, ``if drake_system.dt ==
    1/60`` and ``framerate == 11``, the true video framerate will be::

        max(round(60/11), 1) == 5.

    Args:
        drake_system: System associated with provided trajectory.
        x_trajectory: (T, drake_system.space.n_x) state trajectory.
        framerate: desired frames per second of output video.

    Returns:
        (1, T, 3, H, W) ndarray video capture of trajectory with resolution
        H x W, which are set to 480x640 in :py:mod:`dair_pll.drake_utils`.
        The true framerate, rounded to an integer.

    Todo:
        Only option for implementation at the moment is to access various
        protected members of :py:class:`pydrake.visualization.VideoWriter`\ .
        This function should be updated as `pydrake` has this functionality
        properly exposed.
    """
    assert drake_system.plant_diagram.visualizer is not None
    assert x_trajectory.dim() == 2
    # pylint: disable=protected-access

    vis = drake_system.plant_diagram.visualizer
    sim = drake_system.plant_diagram.sim

    # Downsample trajectory to approximate framerate.
    temporal_downsample = max(round((1 / drake_system.dt) / framerate), 1)
    actual_framerate = round((1 / drake_system.dt) / temporal_downsample)
    x_trajectory = x_trajectory[::temporal_downsample, :]

    # Clear the images before iterating through the trajectory (by default the
    # video starts with one image of the systems at the origin).
    if isinstance(vis, list):
        for v in vis:
            v._pil_images = []
    else:
        vis._pil_images = []  # type: ignore

    # Simulate the system according to the provided data.
    _, carry = drake_system.sample_initial_condition()
    for x_current in x_trajectory:
        drake_system.preprocess_initial_condition(x_current.unsqueeze(0), carry)

        # Force publish video frame.
        sim_context = sim.get_mutable_context()
        if isinstance(vis, list):
            for v in vis:
                video_context = v.GetMyContextFromRoot(sim_context)
                v._publish(video_context)
        else:
            video_context = vis.GetMyContextFromRoot(sim_context)
            vis._publish(video_context)

    # Compose a video ndarray of shape (T, H, W, 4[rgba]).
    if isinstance(vis, list):
        video = []
        for v in vis:
            video.append(np.stack([np.asarray(frame) for frame in v._pil_images]))
            v.Save()    # save to visualizer._filename
    else:
        video = np.stack([np.asarray(frame) for frame in vis._pil_images
                        ])  # type: ignore
        vis.Save()  # save to visualizer._filename

    # Since Drake's VideoWriter defaults to not looping gifs, re-load and re-
    # save the gif to ensure it loops.  This gif is only for debugging purposes,
    # as the gif gets overwritten with every trajectory.  The actual output of
    # this function is a numpy array.
    # filename_postfix is used here because the same visualizer may be used to 
    # visualize different trajectories through here, but they share a v._filename. 
    # With filename_postfix, we can differentiate them.
    if isinstance(vis, list):
        for v in vis:
            vizualization_image = Image.open(v._filename)
            new_name = v._filename.split('.')[0] + filename_postfix + '_.gif'
            vizualization_image.save(new_name, save_all=True, loop=0)
            vizualization_image.close()
    else:
        vizualization_image = Image.open(vis._filename)  # type: ignore
        new_name = vis._filename.split('.')[0] +filename_postfix + '_.gif'  # type: ignore
        vizualization_image.save(new_name, save_all=True, loop=0)
        vizualization_image.close()

    # Remove alpha channel and reorder axes to output type.
    # video: (T, H, W, 4) -> (1, T, 3, H, W)
    if isinstance(vis, list):
        video = [np.expand_dims(np.moveaxis(v, 3, 1), 0)[:, :, :3, :, :] for v in video]
        if filename_postfix == 'geometry':
            video = video[1] # no need to concatenate different views (the three views are left, front, bottom)
        else:
            video = np.concatenate(video, axis=-1)
    else:
        video = np.expand_dims(np.moveaxis(video, 3, 1), 0)
        video = video[:, :, :3, :, :]
    return video, actual_framerate

### For visualizing loss trajectories
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
import os.path as op
from copy import deepcopy
from tqdm import tqdm
def create_plot(data, frame_index, title=None):
    ### Find the relative index of frame_index in the data x. 
    ### If frame_index < x.min(), then the plot will be empty.
    ### If frame_index > x.max(), then the plot will be the same as the last frame.

    x = data['x']

    fig, ax = plt.subplots(figsize=(6, 4.5))
    if frame_index < x.min():
        pass
    else:
        ### Find the index of the frame_index in the data x, 
        ### such that x[index] <= frame_index < x[index+1]
        index = np.searchsorted(x, frame_index, side='right') - 1
        len_indiced = index + 1

        y_keys = list(data.keys())
        y_keys = [y_key for y_key in y_keys if y_key != 'x']
        y_keys.sort()
        x = deepcopy(data['x'][:len_indiced])
        y_dict = {y: deepcopy(data[y][:len_indiced]) for y in y_keys}
        y_arrays = [y_dict[y] for y in y_keys]
        if len(y_arrays[0]) == 1:
            ### Plot with marker because the line won't be visible
            for y in y_keys:
                ax.plot(x, y_dict[y], marker='o', label=y)
        else:
            for y in y_keys:
                ax.plot(x, y_dict[y], label=y)
        if x.min() == x.max():
            ax.set_xlim(x.min(), x.max() + 1)
        else:
            ax.set_xlim(x.min(), x.max())
        ax.set_ylim(min(y.min() for y in y_arrays), max(y.max() for y in y_arrays))
        ax.legend()

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('Frame')
    ### Use x tick minimal interval to 1
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    plot_image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
    plot_image = plot_image.reshape(canvas.get_width_height()[::-1] + (4,))
    plot_image = plot_image[:, :, :3]
    plt.close(fig)
    return plot_image

def sync_video_and_plots(video_rgb, video_drake, data_list, title_list, 
                         output_path, learned_system):
    '''
    video_drake: ndarray of shape 1 * T * 3 * H * W or a string path to a gif or mp4 file'''
    fps_original = 30
    fps_downsample_rate = 30
    fps = fps_original // fps_downsample_rate
    width = 640
    height = 480
    rgb_available = os.path.exists(video_rgb)
    print("Reading rgb video...")
    if rgb_available:
        frames_rgb = read_video(video_rgb)
    else:
        print("RGB video not available. Using black frames.")
    
    if isinstance(video_drake, str):
        print("Reading drake video...")
        frames_drake = read_video(video_drake)
    else:
        frames_drake = video_drake

    n_plots_loss = len(data_list)
    
    # Determine the number of columns and rows of loss plots
    cols_loss = (n_plots_loss + 1) // 2
    rows_loss = 2 if n_plots_loss > 1 else 1
    
    # Resize and arrange the loss plots
    plot_height = height // rows_loss
    plot_width = width // rows_loss

    # grad plots
    flag_plot_grad = learned_system.vis_hook.enabled
    n_plots_grad = len(learned_system.vis_hook.loss_grad_to_vis) if flag_plot_grad else 0

    grad_no_second_row = n_plots_grad == 1
    grad_fill_first_row = n_plots_grad == 1 or n_plots_grad >= 5
    
    # Determine the number of columns and rows considering 
    # the relationship between loss and grad plots
    cols_first_row = cols_loss + 2 if grad_fill_first_row else cols_loss + 1

    if grad_no_second_row:
        cols_second_row = 0
    elif grad_fill_first_row:
        # Put one grad plot at the end of first row
        cols_second_row = max(cols_first_row, n_plots_grad-1)
    else:
        # Put all plots in the second row
        cols_second_row = max(cols_first_row, n_plots_grad)

    cols_first_row = max(cols_first_row, cols_second_row)

    # Make sure the output path's dir is a directory
    if not os.path.isdir(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    tqdm.write("Processing video...")
    ### TODO: not sure if the drake video are of the x_plus or x_current
    ### current implementation assumes x_plut

    if not rgb_available:
        if isinstance(frames_drake, np.ndarray):
            frames_rgb = np.zeros((frames_drake.shape[1], height//2, width//2, 3), dtype=np.uint8)
        else:
            frames_rgb = np.zeros((len(frames_drake), height//2, width//2, 3), dtype=np.uint8)
    
    frames_idxs = np.arange(len(frames_rgb), step=fps_downsample_rate)
    frame_count = 0
    with TemporaryDirectory(prefix="synced-slice-") as tmpdir:
        print(f'Storing temporary files at {tmpdir}')
        for i in tqdm(frames_idxs):

            output_first_row = np.zeros((height, plot_width * cols_first_row, 3), dtype=np.uint8)
            
            if not rgb_available:
                frame_rgb = frames_rgb[i]
            else:
                # if isinstance(frames_rgb[i], str):
                #     frame_rgb_file = frames_rgb[i]
                #     frame_rgb = cv2.imread(op.join(video_rgb, frame_rgb_file))
                #     frame_rgb = cv2.resize(frame_rgb, (width // 2, height // 2))
                #     # bgr to rgb
                #     frame_rgb = frame_rgb[:, :, ::-1]
                # else:
                frame_rgb = frames_rgb[i]
                frame_rgb = cv2.resize(frame_rgb, (width // 2, height // 2))
                
            output_first_row[:height // 2, :width//2] = frame_rgb
                
            x_start = data_list[0]['x'].min() # 1 indexed
            if not rgb_available:
                assert x_start == 1, "x_start should be 1 (no offset) if rgb video is not available"
            i_in_video_drake = i - x_start
            i_in_video_drake = min(i_in_video_drake, len(frames_drake) - 1)
            i_in_video_drake = max(i_in_video_drake, 0)
            
            if isinstance(frames_drake, np.ndarray):
                frame_drake = frames_drake[0, i_in_video_drake] # 3 * H * W
                frame_drake = np.transpose(frame_drake, (1, 2, 0))
            else:
                frame_drake = frames_drake[i_in_video_drake]
                
            frame_drake = cv2.resize(frame_drake, (width // 2, height // 2))

            output_first_row[height//2:, :width//2] = frame_drake

            ### load plots of loss trajectories
            plots = []
            for data, title in zip(data_list, title_list):
                plot = create_plot(data, i, title)
                plots.append(plot)
            
            for i_loss, plot in enumerate(plots):
                resized_plot = cv2.resize(plot, (plot_width, plot_height))
                row = i_loss % rows_loss
                col = i_loss // rows_loss + 1
                output_first_row[row * plot_height:(row + 1) * plot_height, col * plot_width:(col + 1) * plot_width] = resized_plot

            output_frame = output_first_row

            if flag_plot_grad:
                ### load plots of gradients
                plots_grad = dict()
                figs = learned_system.vis_hook.visualize_single_frame_grads_by_losses(i_in_video_drake)

                for key, fig in figs.items():
                    fig.canvas.draw()
                    plot_image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
                    plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                    plot_image = plot_image[:, :, :3]
                    plots_grad[key] = plot_image
                    plt.close(fig)

                i_grad = 0
                keys = list(plots_grad.keys())
                keys.sort()
                key_to_fill_first_row = None
                if grad_fill_first_row:
                    key_to_fill_first_row = 'all' if 'all' in keys else keys[0]
                if not grad_no_second_row:
                    output_second_row = np.zeros((height, plot_width * cols_second_row, 3), dtype=np.uint8)
                
                for key in keys:
                    plot = plots_grad[key]
                    resized_plot = cv2.resize(plot, (plot_width, height))

                    if key == key_to_fill_first_row:
                        output_first_row[:, (cols_loss+1) * plot_width : (cols_loss+2) * plot_width] = resized_plot
                    else:
                        assert not grad_no_second_row
                        output_second_row[:, i_grad * plot_width:(i_grad + 1) * plot_width] = resized_plot
                        i_grad += 1
                
                if grad_no_second_row:
                    output_frame = output_first_row
                else:
                    output_frame = np.vstack((output_first_row, output_second_row))

            # Save the output_frame to a temporary png file
            # Do not using cv2.imwrite because it will compress the image
            # and the quality will be lower. Video size at the end similar. 
            # Efficiency also similar. But opencv video can be opened by Windows Media Player.
            # (opencv video converted by ffmpeg is slightly larger)
            Image.fromarray(output_frame).save(op.join(tmpdir, f'{frame_count:}.png'))
            frame_count += 1
        
        ### run ffmpeg to convert the tmp png files to mp4
        os.system(f"ffmpeg -y -r {fps} -i {tmpdir}/%d.png -vcodec libx264 {output_path}")
        # To make the video openable by Windows Media Player, use -pix_fmt yuv420p, 
        # but the quality is lower. 
        # To use higher quality, use -preset slow -crf 18 or even smaller crf.
