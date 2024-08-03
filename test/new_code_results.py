"""Test to compare the results of the new code with the old code."""

import os.path as op
import pdb
import torch

from dair_pll import file_utils


DEBUG_NEW_CODE_ON_OLD_DATA = False
DEBUG_NEW_CODE_ON_NEW_DATA = True


def get_contact_results(run_name):
    output_dir = file_utils.geom_for_bsdf_dir(STORAGE_DIR, run_name)

    dirs = torch.load(op.join(output_dir, 'support_directions.pt'),
        weights_only=True)
    forces = torch.load(op.join(output_dir, 'support_point_normal_forces.pt'),
        weights_only=True)
    pts = torch.load(op.join(output_dir, 'support_points.pt'),
        weights_only=True)
    states = torch.load(op.join(output_dir, 'support_point_states.pt'),
        weights_only=True)
    toss_frames = torch.load(op.join(output_dir, 'tosses_and_frames.pt'),
        weights_only=True)

    return dirs, forces, pts, states, toss_frames


# First debug PLL on our old toss data with the robot
# interaction changes.
if DEBUG_NEW_CODE_ON_OLD_DATA:
    STORAGE_DIR = op.join(
        file_utils.RESULTS_DIR, 'vision_cube/cube_1/bundlesdf_iteration_1/')

    NEW_RESULTS_RUN_NAME = 'pll_id_00-robot'
    OLD_RESULTS_RUN_NAME = 'pll_id_00-nrobot'
    FIX_RESULTS_RUN_NAME = 'pll_id_x09'

    INPUT_DATA = torch.load(
        file_utils.get_asset(
            'vision_cube/cube_1/toss/bundlesdf_iteration_1/bundlesdf_id_00/1.pt'),
        weights_only=True)

    TACTILE_DATA = torch.load('/home/bibit/Downloads/0_tactile.pt')

    new_dirs, new_forces, new_pts, new_states, new_toss_frames = \
        get_contact_results(NEW_RESULTS_RUN_NAME)
    old_dirs, old_forces, old_pts, old_states, old_toss_frames = \
        get_contact_results(OLD_RESULTS_RUN_NAME)
    fix_dirs, fix_forces, fix_pts, fix_states, fix_toss_frames = \
        get_contact_results(FIX_RESULTS_RUN_NAME)
    
# Next debug the new code on new robot interaction data.
if DEBUG_NEW_CODE_ON_NEW_DATA:
    TACTILE_DATA = torch.load('/home/bibit/Downloads/0_tactile.pt')

    # This remote data had an issue where the toss and full trajectories were
    # off by 1.
    remote_toss_traj = torch.load(
        file_utils.get_asset(
            op.join('vision_robot_bakingbox_sticky_A_REMOTE',
                    'robot_bakingbox_sticky_A_1',
                    'toss/bundlesdf_iteration_1/bundlesdf_id_00/1.pt')),
        )  #weights_only=True)
    remote_full_traj = torch.load(
        file_utils.get_asset(
            op.join('vision_robot_bakingbox_sticky_A_REMOTE',
                    'robot_bakingbox_sticky_A_1',
                    'full/bundlesdf_iteration_1/bundlesdf_id_00.pt')),
        )  #weights_only=True)

    # This local data fixed the issue.
    toss_traj = torch.load(
        file_utils.get_asset(
            op.join('vision_robot_bakingbox_sticky_A',
                    'robot_bakingbox_sticky_A_2',
                    'toss/bundlesdf_iteration_1/bundlesdf_id_00/2.pt')),
        )  #weights_only=True)
    full_traj = torch.load(
        file_utils.get_asset(
            op.join('vision_robot_bakingbox_sticky_A',
                    'robot_bakingbox_sticky_A_2',
                    'full/bundlesdf_iteration_1/bundlesdf_id_00.pt')),
        )  #weights_only=True)


pdb.set_trace()
