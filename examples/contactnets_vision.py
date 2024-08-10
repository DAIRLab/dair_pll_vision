
"""Simple ContactNets/differentiable physics learning examples."""
# pylint: disable=E1103
import os
import os.path as op
import pdb
import time
from typing import cast, Tuple

import click
from torch import Tensor

from dair_pll import file_utils
from dair_pll.data_config import TrajectorySliceConfig
from dair_pll.vision_config import VisionDataConfig, VisionExperiment, \
    VisionExperimentConfig, VisionRobotExperiment, VISION_SYSTEMS, \
    VISION_CUBE_SYSTEM, VISION_PRISM_SYSTEM, VISION_TOBLERONE_SYSTEM, \
    VISION_MILK_SYSTEM
from dair_pll.drake_experiment import DrakeSystemConfig, \
    MultibodyLearnableSystemConfig, MultibodyLosses
from dair_pll.experiment import default_epoch_callback
from dair_pll.experiment_config import OptimizerConfig
from dair_pll.hyperparameter import Float, Int
from dair_pll.multibody_learnable_system import MultibodyLearnableSystem
from dair_pll.multibody_terms import InertiaLearn
from dair_pll.system import System



# File management.
CUBE_MESH_URDF_ASSET = 'bundlesdf_cube_mesh.urdf'
PRISM_MESH_URDF_ASSET = 'bundlesdf_prism_mesh.urdf'
TOBLERONE_MESH_URDF_ASSET = 'bundlesdf_toblerone_mesh.urdf'
MILK_MESH_URDF_ASSET = 'bundlesdf_milk_mesh.urdf'
# Franka URDF for robot experiments.
FRANKA_URDF_ASSET = 'franka_with_ee.urdf'

MESH_TYPE = 'mesh'
CUBE_URDFS = {MESH_TYPE: CUBE_MESH_URDF_ASSET}
PRISM_URDFS = {MESH_TYPE: PRISM_MESH_URDF_ASSET}
TOBLERONE_URDFS = {MESH_TYPE: TOBLERONE_MESH_URDF_ASSET}
MILK_URDFS = {MESH_TYPE: MILK_MESH_URDF_ASSET}
URDFS = {VISION_CUBE_SYSTEM: CUBE_URDFS,
         VISION_PRISM_SYSTEM: PRISM_URDFS,
         VISION_TOBLERONE_SYSTEM: TOBLERONE_URDFS,
         VISION_MILK_SYSTEM: MILK_URDFS}

ROBOT_CONSTANT_BODIES = [
    'panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4',
    'panda_link5', 'panda_link6', 'panda_link7', 'end_effector_base',
    'end_effector_link', 'end_effector_tip'
]

# Data configuration.
DT = 0.0333 #0.0068 # 1/frame rate of the camera

# Training data configuration.
T_PREDICTION = 1

# Optimization configuration.
CUBE_LR = 1e-3
PRISM_LR = 1e-3
TOBLERONE_LR = 1e-3
MILK_LR = 1e-3
LRS = {VISION_CUBE_SYSTEM: CUBE_LR,
       VISION_PRISM_SYSTEM: PRISM_LR,
       VISION_TOBLERONE_SYSTEM: TOBLERONE_LR,
       VISION_MILK_SYSTEM: MILK_LR, 
       'vision_bottle': 1e-3,
       'vision_half': 1e-3,
       'vision_egg': 1e-3,
       'vision_napkin': 1e-3,
       'vision_bakingbox': 1e-3,
       'vision_burger': 1e-3,
       'vision_cardboard': 1e-3,
       'vision_chocolate': 1e-3,
       'vision_cream': 1e-3,
       'vision_croc': 1e-3,
       'vision_crushedcan': 1e-3,
       'vision_duck': 1e-3,
       'vision_gallon': 1e-3,
       'vision_greencan': 1e-3,
       'vision_hotdog': 1e-3,
       'vision_icetray': 1e-3,
       'vision_mug': 1e-3,
       'vision_oatly': 1e-3,
       'vision_pinkcan': 1e-3,
       'vision_stapler': 1e-3,
       'vision_styrofoam': 1e-3,
       'vision_toothpaste': 1e-3,
       'vision_robot_bakingbox_sticky_A': 1e-3,
       }
CUBE_WD = 0.0
PRISM_WD = 0.0
TOBLERONE_WD = 0.0
MILK_WD = 0.0
WDS = {VISION_CUBE_SYSTEM: CUBE_WD,
       VISION_PRISM_SYSTEM: PRISM_WD,
       VISION_TOBLERONE_SYSTEM: TOBLERONE_WD,
       VISION_MILK_SYSTEM: MILK_WD,
       'vision_bottle': 0.0,
       'vision_half': 0.0,
       'vision_egg': 0.0,
       'vision_napkin': 0.0,
       'vision_bakingbox': 0.0,
       'vision_burger': 0.0,
       'vision_cardboard': 0.0,
       'vision_chocolate': 0.0,
       'vision_cream': 0.0,
       'vision_croc': 0.0,
       'vision_crushedcan': 0.0,
       'vision_duck': 0.0,
       'vision_gallon': 0.0,
       'vision_greencan': 0.0,
       'vision_hotdog': 0.0,
       'vision_icetray': 0.0,
       'vision_mug': 0.0,
       'vision_oatly': 0.0,
       'vision_pinkcan': 0.0,
       'vision_stapler': 0.0,
       'vision_styrofoam': 0.0,
       'vision_toothpaste': 0.0,
       'vision_robot_bakingbox_sticky_A': 0.0,
}
EPOCHS = 200 #500
PATIENCE = 100 #EPOCHS

WANDB_PROJECT = 'dair_pll-vision'

SKIP_VIDEO_OPTIONS = ['none', 'all', 'geometry', 'rollout']
LEARN_INERTIA_OPTIONS = ['none', 'all']     # May want to add more cases, e.g.
                                            # learn CoM only.

DRAKE_PYTORCH_FUNCTION_EXPORT_DIR = '/home/minghz/Desktop/symbolic_pytorch_v2'

# Loss term weights.
DEFAULT_W_PRED = 1.0
DEFAULT_W_COMP = 1.0
DEFAULT_W_DISS = 1.0
DEFAULT_W_PEN = 20.0
DEFAULT_W_BSDF = 0.02


def get_storage_names(system: str, start_toss: int, end_toss: int,
                      cycle_iteration: int) -> Tuple[str, str, str]:
    """Using the expected file structure designed for the vision experiments,
    return the asset folders name, tracker name, and storage folders name.
    
    Args:
        system: Which system to learn.
        start_toss: start toss number of data to load.
        end_toss: end toss number of data to load.
        cycle_iteration: BundleSDF iteration number (0 means use TagSLAM poses).

    Returns:
        asset_name:  Name of the asset folder, e.g. 'cube_2'
        tracker:  Name of the tracker folder, e.g. 'bundlesdf_iteration_1'.
        storage_name: Name of the storage folder, e.g.
            'vision_cube/cube_2/bundlesdf_iteration_1'.
    """
    asset_name = '_'.join(system.split('_')[1:]) + f'_{start_toss}'
    asset_name += f'-{end_toss}' if start_toss != end_toss else ''
    tracker = 'tagslam' if cycle_iteration == 0 else \
        f'bundlesdf_iteration_{cycle_iteration}'
    return asset_name, tracker, op.join(system, asset_name, tracker)


def make_urdf_with_bundlesdf_mesh(
        system: str, vision_asset: str, bundlesdf_id: str,
        cycle_iteration: int, storage_name: str, pll_id: str) -> str:
    """To use BundleSDF mesh as geometry for comparison, copy the mesh from the
    BundleSDF outputs in the PLL assets directory to the PLL run directory along
    with a URDF template that will reference the mesh.  Returns the path to the
    created URDF file."""
    if not bundlesdf_id.startswith('bundlesdf_id_'):
        bundlesdf_id = f'bundlesdf_id_{bundlesdf_id}'

    asset_subdirs = op.join(system, vision_asset)
    bundlesdf_mesh_filepath = file_utils.get_mesh_from_bundlesdf(
        asset_subdirs, bundlesdf_id, cycle_iteration)
    urdf_template_filepath = file_utils.get_vision_urdf_template_path()

    target_dir = file_utils.run_dir(storage_name, pll_id)
    target_mesh_filepath = op.join(target_dir, 'bundlesdf_mesh.obj')
    target_urdf_filepath = op.join(target_dir, 'with_bundlesdf_mesh.urdf')

    os.system(f'cp {bundlesdf_mesh_filepath} {target_mesh_filepath}')
    os.system(f'cp {urdf_template_filepath} {target_urdf_filepath}')

    return target_urdf_filepath

    
def main(pll_run_id: str = "",
         system: str = VISION_CUBE_SYSTEM,
         start_toss: int = 2,
         end_toss: int = 2,
         cycle_iteration: int = 1,
         bundlesdf_id: str = None,
         contactnets: bool = True,
         regenerate: bool = False,
         pretrained_icnn_weights_filepath: str = None,
         learn_inertia: str = 'all',
         skip_videos: str = 'rollout',
         clear_data: bool = False,
         w_pred: float = DEFAULT_W_PRED,
         w_comp: float = DEFAULT_W_COMP,
         w_diss: float = DEFAULT_W_DISS,
         w_pen: float = DEFAULT_W_PEN,
         w_bsdf: float = DEFAULT_W_BSDF,
         use_bundlesdf_mesh: bool = True,
         is_robot_experiment: bool = False,
         export_drake_pytorch: bool = False):
    """Execute ContactNets basic example on a system.

    Args:
        pll_run_id: name of experiment run.
        system: Which system to learn.
        cycle_iteration: BundleSDF iteration number (0 means use TagSLAM poses).
        contactnets: Whether to use ContactNets or prediction loss
        regenerate: Whether save updated URDF's each epoch.
        pretrained_icnn_weights_filepath: Filepath to set of pretrained
          ICNN weights.
        learn_inertia: What inertia parameters to learn (none, all); more
          options to be implemented.  Also note that 'all' actually means 9
          parameters (6 moments/products of inertia, CoM location) and excludes
          the mass itself, which is unobservable.
        skip_videos: What videos to skip generating at every epoch; can be
          'none', 'all', 'rollout' (default), or 'geometry'.  Skipping
          generating all of these saves a lot of time (10x or more speedup).
        clear_data: Whether to clear storage folder before running.
        w_pred: Weight of prediction loss term.
        w_comp: Weight of complimentarity loss term.
        w_diss: Weight of dissipation loss term.
        w_pen: Weight of penetration loss term.
        w_bsdf: Weight of BundleSDF loss term.
        use_bundlesdf_mesh: Whether to use the mesh from BundleSDF or the
          default URDF for the system (warning: the default URDFs are not
          origin-aligned to the BundleSDF tracking origin).
        is_robot_experiment: Whether this is a robot experiment.
        export_drake_pytorch: Whether to export Drake Pytorch expressions to
          files for reuse in later experiments.  If true, this experiment will
          terminate after the export has completed.
    """
    # pylint: disable=too-many-locals, too-many-arguments
    if pll_run_id == "":
        pll_run_id = f'pll_id_{str(int(time.time()))}'
    elif pll_run_id[:7] != 'pll_id_':
        pll_run_id = f'pll_id_{pll_run_id}'

    print(f'\n\tCreating (or restarting) run: {pll_run_id}' \
         + f'\n\ton system: {system}' \
         + f'\n\twith BundleSDF cycle iteration: {cycle_iteration}' \
         + f'\n\tusing ContactNets: {contactnets}' \
         + f'\n\tregenerate: {regenerate}' \
         + f'\n\twith pretrained ICNN weights at: ' \
         + f'{pretrained_icnn_weights_filepath}' \
         + f'\n\tlearning inertia: {learn_inertia}' \
         + f'\n\tclear_data: {clear_data}' \
         + f'\n\tw_pred: {w_pred}' \
         + f'\n\tw_comp: {w_comp}' \
         + f'\n\tw_diss: {w_diss}' \
         + f'\n\tw_pen: {w_pen}' \
         + f'\n\tw_bsdf: {w_bsdf}' \
         + f'\n\tusing BundleSDF mesh: {use_bundlesdf_mesh}' \
         + f'\n\tskipping video generation: {skip_videos}' \
         + f'\n\tand is robot experiment: {is_robot_experiment} \n')
    
    # First step, clear out data on disk for a fresh start.
    asset_name, tracker, storage_name = get_storage_names(
        system, start_toss, end_toss, cycle_iteration)
    run_directory = file_utils.run_dir(storage_name, pll_run_id, create=False)
    if op.exists(run_directory):        
        if clear_data:
            os.system(f'rm -r {run_directory}')
        else:
            print(f'Directory {run_directory} already exists and not set to ' \
                  + f'clear (run with --clear-data next time).  Exiting.')
            exit()

    # Next, build the configuration of the learning experiment.

    # Describes the optimizer settings; by default, the optimizer is Adam.
    optimizer_config = OptimizerConfig(lr=Float(LRS[system]),
                                       wd=Float(WDS[system]),
                                       patience=PATIENCE,
                                       epochs=EPOCHS)

    # Describes the ground truth system; infers everything from the URDFs.
    # This is a configuration for a DrakeSystem, which wraps a Drake simulation
    # for the described URDFs.
    if use_bundlesdf_mesh and tracker != 'tagslam' and w_bsdf > 0:
        urdf = make_urdf_with_bundlesdf_mesh(
            system=system, vision_asset=asset_name, bundlesdf_id=bundlesdf_id,
            cycle_iteration=cycle_iteration, storage_name=storage_name,
            pll_id=pll_run_id
        )
    else:
        urdf_asset = URDFS[VISION_CUBE_SYSTEM][MESH_TYPE]
        urdf = file_utils.get_asset(urdf_asset)
    urdfs = {'object': urdf}
    if is_robot_experiment:
        urdfs['robot'] = file_utils.get_asset(FRANKA_URDF_ASSET)
    base_config = DrakeSystemConfig(urdfs=urdfs)

    # If this is a robot experiment, use precomputed mass matrix and lagrangian
    # forces functions.
    if is_robot_experiment:
        txt_function_directory = \
            file_utils.get_asset('precomputed_vision_functions')
        precomputed_function_directories = {
            'mass_matrix': txt_function_directory,
            'lagrangian_forces': txt_function_directory}
    else:
        precomputed_function_directories = {}

    # Describes the learnable system. The MultibodyLearnableSystem type learns
    # a multibody system, which is initialized as the system in the given URDFs.
    # Don't learn the mass even if learning the rest of the inertia.
    # TODO could reconsider this for robot experiments.
    learnable_config = MultibodyLearnableSystemConfig(
        urdfs=urdfs,
        loss = MultibodyLosses.VISION_LOSS if contactnets else \
            MultibodyLosses.PREDICTION_LOSS,
        constant_bodies = ROBOT_CONSTANT_BODIES if is_robot_experiment else [],
        inertia_mode = InertiaLearn(
            mass=False, com=learn_inertia=='all', inertia=learn_inertia=='all'),
        pretrained_icnn_weights_filepath=pretrained_icnn_weights_filepath,
        w_pred=w_pred, w_comp=w_comp, w_diss=w_diss, w_pen=w_pen,
        w_bsdf=w_bsdf, represent_geometry_as='mesh',
        precomputed_function_directories=precomputed_function_directories,
        export_drake_pytorch_dir = DRAKE_PYTORCH_FUNCTION_EXPORT_DIR if \
            export_drake_pytorch else None
    )

    # How to slice trajectories into training datapoints.
    if is_robot_experiment:
        previous_state_keys = ['object_state', 'robot_state', 'robot_effort']
        future_state_keys = ['object_state', 'robot_state']
    else:
        previous_state_keys = ['state']
        future_state_keys = ['state']
    slice_config = TrajectorySliceConfig(
        t_prediction=1 if contactnets else T_PREDICTION,
        his_state_keys=previous_state_keys,
        pred_state_keys=future_state_keys)

    # Describes configuration of the data.
    data_config = VisionDataConfig(
        dt=DT,
        train_fraction=0.7,
        valid_fraction=0.3,
        test_fraction=0.0,
        slice_config=slice_config,
        update_dynamically=False,
        asset_subdirectories=op.join(system, asset_name),
        tracker=tracker,
        bundlesdf_id=bundlesdf_id)

    # Combines everything into config for entire experiment.
    gen_geom_videos = False if skip_videos in ['all', 'geometry'] else True
    gen_pred_videos = False if skip_videos in ['all', 'rollout'] else True
    experiment_config = VisionExperimentConfig(
        storage=file_utils.storage_dir(storage_name),
        run_name=pll_run_id,
        base_config=base_config,
        learnable_config=learnable_config,
        optimizer_config=optimizer_config,
        data_config=data_config,
        full_evaluation_period=1,
        update_geometry_in_videos=True,
        generate_video_predictions_throughout=gen_pred_videos,
        generate_video_geometries_throughout=gen_geom_videos,
        run_wandb=True,
        wandb_project=WANDB_PROJECT
    )

    # Makes experiment.
    print('Making experiment.')
    experiment = VisionRobotExperiment(experiment_config) if \
        is_robot_experiment else VisionExperiment(experiment_config)

    # No need to prepare data for vision experiments since all assets from the
    # asset directory are used.

    def regenerate_callback(epoch: int, learned_system: System,
                            train_loss: Tensor,
                            best_valid_loss: Tensor) -> None:
        default_epoch_callback(epoch, learned_system, train_loss,
                               best_valid_loss)
        cast(MultibodyLearnableSystem, learned_system).generate_updated_urdfs()
    
    # Trains system and saves final results.
    print(f'Training the model.')
    learned_system, _stats = experiment.generate_results(
        regenerate_callback if regenerate else default_epoch_callback)

    # Save the final urdf.
    print(f'\nSaving the final learned URDF...', end=' ')
    learned_system = cast(MultibodyLearnableSystem, learned_system)
    learned_system.generate_updated_urdfs(suffix='best')
    print(f'Done!')

    # Export BundleSDF training data.
    print(f'Saving points and directions...', end=' ')
    experiment.generate_bundlesdf_data(learned_system)
    print(f'Done!')

    
@click.command()
@click.option('--run-name', default="")
@click.option('--vision-asset',
              type=str,
              default=None,
              help="directory of the asset folder e.g. cube_2, assumed to " + \
                "be in a vision_{SYSTEM}/ folder; encodes system and tosses.")
@click.option('--cycle-iteration',
              type=int,
              default=1,
              help="BundleSDF iteration number (0 means use TagSLAM poses).")
@click.option('--bundlesdf-id',
              type=str,
              default=None,
              help="what BundleSDF run ID associated with pose outputs to use.")
@click.option('--contactnets/--prediction',
              default=True,
              help="whether to train/test with ContactNets/prediction loss.")
@click.option('--regenerate/--no-regenerate',
              default=False,
              help="whether to save updated URDFs each epoch.")
@click.option('--pretrained',
              type=str,
              default=None,
              help='pretrained weights of Homonogeneous ICNN')
@click.option('--learn-inertia',
              type=click.Choice(LEARN_INERTIA_OPTIONS),
              default='all',
              help="what inertia parameters to learn (none, all).")
@click.option('--export-drake-pytorch/--train',
              type=bool,
              default=False,
              help="whether to export computed drake pytorch expressions " + \
                "then force terminate the code.")
@click.option('--skip-videos',
              type=click.Choice(SKIP_VIDEO_OPTIONS),
              default='rollout',
              help="what videos to skip generating every epoch (saves time)" + \
                " can be 'none', 'all', 'rollout' (default), or 'geometry'.")
@click.option('--clear-data/--keep-data',
              default=False,
              help="whether to clear experiment results folder before running.")
@click.option('--w-pred',
              type=float,
              default=DEFAULT_W_PRED,
              help="weight of prediction loss term.")
@click.option('--w-comp',
              type=float,
              default=DEFAULT_W_COMP,
              help="weight of complimentarity loss term.")
@click.option('--w-diss',
              type=float,
              default=DEFAULT_W_DISS,
              help="weight of dissipation loss term.")
@click.option('--w-pen',
              type=float,
              default=DEFAULT_W_PEN,
              help="weight of penetration loss term.")
@click.option('--w-bsdf',
              type=float,
              default=DEFAULT_W_BSDF,
              help="weight of BundleSDF loss term.")

def main_command(run_name: str, vision_asset: str, cycle_iteration: int,
                 bundlesdf_id: str, contactnets: bool, regenerate: bool,
                 pretrained: str, learn_inertia: str,
                 export_drake_pytorch: bool, skip_videos: str, clear_data: bool,
                 w_pred: float, w_comp: float, w_diss: float, w_pen: float,
                 w_bsdf: float):
    # First decode the system and start/end tosses from the provided asset
    # directory.
    assert '_' in vision_asset, f'Invalid asset directory: {vision_asset}.'
    system = f"vision_{'_'.join(vision_asset.split('_')[:-1])}"
    assert system in VISION_SYSTEMS, f'Invalid system in {vision_asset=}.'

    toss_key = vision_asset.split('_')[-1]
    start_toss = int(toss_key.split('-')[0])
    end_toss = start_toss if '-' not in toss_key else \
        int(toss_key.split('-')[1])
    assert start_toss <= end_toss, f'Invalid toss range: {start_toss} ' + \
        f'-{end_toss} inferred from {vision_asset=}.'

    is_robot_experiment = system.startswith('vision_robot')

    main(pll_run_id=run_name, system=system, start_toss=start_toss,
         end_toss=end_toss, cycle_iteration=cycle_iteration,
         bundlesdf_id=bundlesdf_id, contactnets=contactnets,
         regenerate=regenerate, pretrained_icnn_weights_filepath=pretrained,
         learn_inertia=learn_inertia, skip_videos=skip_videos,
         clear_data=clear_data, w_pred=w_pred, w_comp=w_comp, w_diss=w_diss,
         w_pen=w_pen, w_bsdf=w_bsdf, is_robot_experiment=is_robot_experiment,
         export_drake_pytorch=export_drake_pytorch)



if __name__ == '__main__':
    main_command()  # pylint: disable=no-value-for-parameter
