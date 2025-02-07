"""Simple script to restart an experiment that has already started."""

import click
import os
import os.path as op
from typing import cast, Tuple, Optional
import torch
import matplotlib.pyplot as plt

from dair_pll import file_utils
from dair_pll.experiment import default_epoch_callback
from dair_pll.multibody_learnable_system import MultibodyLearnableSystem
from dair_pll.vision_config import VisionExperiment, VisionExperimentConfig, \
    VISION_CUBE_SYSTEM, VisionRobotExperiment, check_valid_system # VISION_SYSTEMS



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


def main(pll_run_id: str = "",
         system: str = VISION_CUBE_SYSTEM,
         start_toss: int = 2,
         end_toss: int = 2,
         cycle_iteration: int = 1,
         loss_curve: bool = False,
         vis_frame: int = 0,
         epochs: Optional[int] = None,
         force_continue: bool = False):
    """Restart a PLL vision experiment run.

    Args:
        pll_run_id: name of experiment run.
        system: Which system to learn.
        start_toss
        end_toss
        cycle_iteration: BundleSDF iteration number (0 means use TagSLAM poses).
    """
    _asset, _tracker, storage_folder_name = get_storage_names(
        system, start_toss, end_toss, cycle_iteration)
    storage_name = os.path.join(file_utils.RESULTS_DIR, storage_folder_name)

    # Combines everything into config for entire experiment.
    experiment_config = file_utils.load_configuration(storage_name, pll_run_id)
    assert isinstance(experiment_config, VisionExperimentConfig), \
        f'Expected VisionExperimentConfig, got {type(experiment_config)}.'
    print(f'Loaded original experiment configuration.')

    if epochs is not None:
        experiment_config.optimizer_config.epochs = epochs
        print(f'Overriding epochs to {epochs}.')
    if force_continue:
        experiment_config.optimizer_config.patience = -1
        checkpoint_filename = file_utils.get_model_filename(storage_name, pll_run_id)
        # Load the checkpoint
        checkpoint_dict = torch.load(checkpoint_filename)
        # Modify the finished_training flag
        checkpoint_dict['finished_training'] = False
        # Save it back
        torch.save(checkpoint_dict, checkpoint_filename)
        print('Forcing continuation of training: ' + \
              'set patience to -1 and finished_training to False.')

    is_robot_experiment = system.startswith('vision_robot')
    # Makes experiment.
    experiment = VisionRobotExperiment(experiment_config) if is_robot_experiment \
        else VisionExperiment(experiment_config)


    # Trains system and saves final results.
    print(f'\nTraining the model.')
    # force_evaluation draws the wandb gifs and full loss_traj videos, 
    # which should be done at the end of the continued training,
    # but is not needed if we are only plotting single-frame loss gradient plots.
    learned_system, _stats = experiment.generate_results(default_epoch_callback, 
                                                         force_evaluation=not loss_curve)

    if loss_curve:
        # learned_system, optimizer, training_state = experiment.setup_training_no_wandb()
        # # true_geom_system = experiment.get_true_geometry_multibody_learnable_system()
        # exp_path = op.join(storage_name, pll_run_id)
        # # experiment.loss_over_trajectory(true_geom_system, exp_path, True)
        # # experiment.loss_over_trajectory(learned_system, exp_path, False)
        figs = learned_system.vis_hook.visualize_single_frame_grads_by_losses(vis_frame)
        ### Show all figs
        for loss_name, fig in figs.items():
            plt.figure(fig.number)
            plt.show(block=False)
        input("Press Enter to close all plots...")

        return
    
    # Save the final urdf.
    print(f'\nSaving the final learned URDF.')
    learned_system = cast(MultibodyLearnableSystem, learned_system)
    learned_system.generate_updated_urdfs('best')
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
@click.option('--loss-curve',
              is_flag=True,
              help="Plot the loss curve only.")
@click.option('--vis-frame',
                type=int,
                default=0,
                help="Frame number to visualize losses for.")
@click.option('--epochs', '-e',
                type=int,
                default=None,
                help="Number of epochs to train for.")
@click.option('--force-continue', '-f',
                is_flag=True,
                help="Continue training from the last checkpoint, " + \
                    "regardless of whether the run is finished.")
### Usage: 
# 1. Visualize the loss gradient of a saved model interactively: 
# python restart_vision_run.py --run-name a --vision-asset b --cycle-iteration c --loss-curve --vis-frame d
# 2. Continue training a model that is accidentally stopped:
# python restart_vision_run.py --run-name a --vision-asset b --cycle-iteration c
# 3. Continue training a model that finished training:
# python restart_vision_run.py --run-name a --vision-asset b --cycle-iteration c --force-continue 
# (optional) --epochs e to override the total number of epochs

def main_command(run_name: str, vision_asset: str, cycle_iteration: int,
                 loss_curve: bool, vis_frame: int, epochs: Optional[int], 
                 force_continue: bool):
    """Executes main function with argument interface."""
    # First decode the system and start/end tosses from the provided asset
    # directory.
    assert '_' in vision_asset, f'Invalid asset directory: {vision_asset}.'
    system = f"vision_{'_'.join(vision_asset.split('_')[:-1])}"
    # assert system in VISION_SYSTEMS or 'robot' in system, f'Invalid system in {vision_asset=}.'
    assert check_valid_system(system), f'Invalid system in {vision_asset=}.'

    toss_key = vision_asset.split('_')[-1]
    start_toss = int(toss_key.split('-')[0])
    end_toss = start_toss if '-' not in toss_key else \
        int(toss_key.split('-')[1])
    assert start_toss <= end_toss, f'Invalid toss range: {start_toss} ' + \
        f'-{end_toss} inferred from {vision_asset=}.'
    
    pll_run_id = run_name
    if not pll_run_id.startswith('pll_id_'):
        pll_run_id = f'pll_id_{pll_run_id}'

    main(pll_run_id, system, start_toss, end_toss, cycle_iteration, loss_curve, vis_frame, 
         epochs, force_continue)


if __name__ == '__main__':
    main_command()  # pylint: disable=no-value-for-parameter
