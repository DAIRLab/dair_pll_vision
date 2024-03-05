
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
    VisionExperimentConfig, VISION_SYSTEMS, VISION_CUBE_SYSTEM, \
    VISION_PRISM_SYSTEM, VISION_TOBLERONE_SYSTEM, VISION_MILK_SYSTEM
from dair_pll.drake_experiment import DrakeSystemConfig, \
    MultibodyLearnableSystemConfig, MultibodyLosses
from dair_pll.experiment import default_epoch_callback
from dair_pll.experiment_config import OptimizerConfig
from dair_pll.hyperparameter import Float, Int
from dair_pll.multibody_learnable_system import MultibodyLearnableSystem
from dair_pll.system import System



# File management.
CUBE_MESH_URDF_ASSET = 'bundlesdf_cube_mesh.urdf'
PRISM_MESH_URDF_ASSET = 'bundlesdf_prism_mesh.urdf'
TOBLERONE_MESH_URDF_ASSET = 'bundlesdf_toblerone_mesh.urdf'
MILK_MESH_URDF_ASSET = 'bundlesdf_milk_mesh.urdf'

MESH_TYPE = 'mesh'
CUBE_URDFS = {MESH_TYPE: CUBE_MESH_URDF_ASSET}
PRISM_URDFS = {MESH_TYPE: PRISM_MESH_URDF_ASSET}
TOBLERONE_URDFS = {MESH_TYPE: TOBLERONE_MESH_URDF_ASSET}
MILK_URDFS = {MESH_TYPE: MILK_MESH_URDF_ASSET}
URDFS = {VISION_CUBE_SYSTEM: CUBE_URDFS,
         VISION_PRISM_SYSTEM: PRISM_URDFS,
         VISION_TOBLERONE_SYSTEM: TOBLERONE_URDFS,
         VISION_MILK_SYSTEM: MILK_URDFS}

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
       VISION_MILK_SYSTEM: MILK_LR}
CUBE_WD = 0.0
PRISM_WD = 0.0
TOBLERONE_WD = 0.0
MILK_WD = 0.0
WDS = {VISION_CUBE_SYSTEM: CUBE_WD,
       VISION_PRISM_SYSTEM: PRISM_WD,
       VISION_TOBLERONE_SYSTEM: TOBLERONE_WD,
       VISION_MILK_SYSTEM: MILK_WD}
EPOCHS = 200 #500
PATIENCE = EPOCHS

WANDB_PROJECT = 'dair_pll-vision'


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
    asset_name = system.split('_')[1] + f'_{start_toss}'
    asset_name += f'-{end_toss}' if start_toss != end_toss else ''
    tracker = 'tagslam' if cycle_iteration == 0 else \
        f'bundlesdf_iteration_{cycle_iteration}'
    return asset_name, tracker, op.join(system, asset_name, tracker)

    
def main(pll_run_id: str = "",
         system: str = VISION_CUBE_SYSTEM,
         start_toss: int = 2,
         end_toss: int = 2,
         cycle_iteration: int = 1,
         bundlesdf_id: str = None,
         contactnets: bool = True,
         regenerate: bool = False,
         pretrained_icnn_weights_filepath: str = None,
         clear_data: bool = False):
    """Execute ContactNets basic example on a system.

    Args:
        pll_run_id: name of experiment run.
        system: Which system to learn.
        cycle_iteration: BundleSDF iteration number (0 means use TagSLAM poses).
        contactnets: Whether to use ContactNets or prediction loss
        regenerate: Whether save updated URDF's each epoch.
        pretrained_icnn_weights_filepath: Filepath to set of pretrained
          ICNN weights.
        clear_data: Whether to clear storage folder before running.
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
         + f'\n\tclear_data: {clear_data}\n')
    
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

    # Describes the ground truth system; infers everything from the URDF.
    # This is a configuration for a DrakeSystem, which wraps a Drake simulation
    # for the described URDFs.
    urdf_asset = URDFS[system][MESH_TYPE]
    urdf = file_utils.get_asset(urdf_asset)
    urdfs = {system: urdf}
    base_config = DrakeSystemConfig(urdfs=urdfs)

    # Describes the learnable system. The MultibodyLearnableSystem type learns
    # a multibody system, which is initialized as the system in the given URDFs.
    loss = MultibodyLosses.CONTACTNETS_LOSS \
        if contactnets else \
        MultibodyLosses.PREDICTION_LOSS
    learnable_config = MultibodyLearnableSystemConfig(
      urdfs=urdfs, loss=loss,
      pretrained_icnn_weights_filepath=pretrained_icnn_weights_filepath
    )

    # How to slice trajectories into training datapoints.
    slice_config = TrajectorySliceConfig(
        t_prediction=1 if contactnets else T_PREDICTION)

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
    experiment_config = VisionExperimentConfig(
        storage=file_utils.storage_dir(storage_name),
        run_name=pll_run_id,
        base_config=base_config,
        learnable_config=learnable_config,
        optimizer_config=optimizer_config,
        data_config=data_config,
        full_evaluation_period=1,
        visualize_learned_geometry=True,
        run_wandb=True,
        wandb_project=WANDB_PROJECT
    )

    # Makes experiment.
    print('Making experiment.')
    experiment = VisionExperiment(experiment_config)
    # pdb.set_trace()

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
    learned_system.generate_updated_urdfs()
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
              help="whether save updated URDF's each epoch.")
@click.option('--pretrained',
              type=str,
              default=None,
              help='pretrained weights of Homonogeneous ICNN')
@click.option('--clear-data/--keep-data',
              default=False,
              help="Whether to clear storage folder before running.")

def main_command(run_name: str, vision_asset: str, cycle_iteration: int,
                 bundlesdf_id: str, contactnets: bool, regenerate: bool,
                 pretrained: str, clear_data: bool):
    # First decode the system and start/end tosses from the provided asset
    # directory.
    assert '_' in vision_asset, f'Invalid asset directory: {vision_asset}.'
    system = f"vision_{vision_asset.split('_')[0]}"
    assert system in VISION_SYSTEMS, f'Invalid system in {vision_asset=}.'

    start_toss = int(vision_asset.split('_')[1].split('-')[0])
    end_toss = start_toss if '-' not in vision_asset else \
        int(vision_asset.split('-')[1])
    assert start_toss <= end_toss, f'Invalid toss range: {start_toss} ' + \
        f'-{end_toss} inferred from {vision_asset=}.'

    main(run_name, system, start_toss, end_toss, cycle_iteration, bundlesdf_id,
         contactnets, regenerate, pretrained, clear_data)



if __name__ == '__main__':
    main_command()  # pylint: disable=no-value-for-parameter
