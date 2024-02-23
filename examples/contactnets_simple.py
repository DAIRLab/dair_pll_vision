"""Simple ContactNets/differentiable physics learning examples."""
# pylint: disable=E1103
import os
import pdb
import time
from typing import cast

import click
from torch import Tensor

from dair_pll import file_utils
from dair_pll.data_config import TrajectorySliceConfig, DataConfig
from dair_pll.drake_experiment import \
    DrakeMultibodyLearnableExperiment, DrakeSystemConfig, \
    MultibodyLearnableSystemConfig, MultibodyLosses, \
    DrakeMultibodyLearnableExperimentConfig
from dair_pll.experiment import default_epoch_callback
from dair_pll.experiment_config import OptimizerConfig
from dair_pll.hyperparameter import Float, Int
from dair_pll.multibody_learnable_system import MultibodyLearnableSystem
from dair_pll.system import System


CUBE_SYSTEM = 'vision_cube'
PRISM_SYSTEM = 'vision_prism'
TOBLERONE_SYSTEM = 'vision_toblerone'
MILK_SYSTEM = 'vision_milk'
SYSTEMS = [CUBE_SYSTEM, PRISM_SYSTEM, TOBLERONE_SYSTEM, MILK_SYSTEM]

# File management.
CUBE_DATA_ASSET = 'vision_cube'
PRISM_DATA_ASSET = 'vision_prism'
TOBLERONE_DATA_ASSET = 'vision_toblerone'
MILK_DATA_ASSET = 'vision_milk'
DATA_ASSETS = {CUBE_SYSTEM: CUBE_DATA_ASSET,
               PRISM_SYSTEM: PRISM_DATA_ASSET,
               TOBLERONE_SYSTEM: TOBLERONE_DATA_ASSET,
               MILK_SYSTEM: MILK_DATA_ASSET}

CUBE_MESH_URDF_ASSET = 'bundlesdf_cube_mesh.urdf'
PRISM_MESH_URDF_ASSET = 'bundlesdf_prism_mesh.urdf'
TOBLERONE_MESH_URDF_ASSET = 'bundlesdf_toblerone_mesh.urdf'
MILK_MESH_URDF_ASSET = 'bundlesdf_milk_mesh.urdf'

MESH_TYPE = 'mesh'
CUBE_URDFS = {MESH_TYPE: CUBE_MESH_URDF_ASSET}
PRISM_URDFS = {MESH_TYPE: PRISM_MESH_URDF_ASSET}
TOBLERONE_URDFS = {MESH_TYPE: TOBLERONE_MESH_URDF_ASSET}
MILK_URDFS = {MESH_TYPE: MILK_MESH_URDF_ASSET}
URDFS = {CUBE_SYSTEM: CUBE_URDFS,
         PRISM_SYSTEM: PRISM_URDFS,
         TOBLERONE_SYSTEM: TOBLERONE_URDFS,
         MILK_SYSTEM: MILK_URDFS}

# Data configuration.
DT = 0.0333 #0.0068 # 1/frame rate of the camera

# Training data configuration.
T_PREDICTION = 1

# Optimization configuration.
CUBE_LR = 1e-3
PRISM_LR = 1e-3
TOBLERONE_LR = 1e-3
MILK_LR = 1e-3
LRS = {CUBE_SYSTEM: CUBE_LR,
       PRISM_SYSTEM: PRISM_LR,
       TOBLERONE_SYSTEM: TOBLERONE_LR,
       MILK_SYSTEM: MILK_LR}
CUBE_WD = 0.0
PRISM_WD = 0.0
TOBLERONE_WD = 0.0
MILK_WD = 0.0
WDS = {CUBE_SYSTEM: CUBE_WD,
       PRISM_SYSTEM: PRISM_WD,
       TOBLERONE_SYSTEM: TOBLERONE_WD,
       MILK_SYSTEM: MILK_WD}
EPOCHS = 200 #500
PATIENCE = EPOCHS

WANDB_PROJECT = 'dair_pll-vision'


def main(run_name: str = "",
         system: str = CUBE_SYSTEM,
         cycle_iteration: int = 1,
         contactnets: bool = True,
         regenerate: bool = False,
         dataset_size: int = 512,
         pretrained_icnn_weights_filepath: str = None,
         clear_data: bool = False):
    """Execute ContactNets basic example on a system.

    Args:
        run_name: name of experiment run.
        system: Which system to learn.
        cycle_iteration: BundleSDF iteration number (0 means use TagSLAM poses).
        contactnets: Whether to use ContactNets or prediction loss
        regenerate: Whether save updated URDF's each epoch.
        dataset_size: Number of trajectories to use.
        pretrained_icnn_weights_filepath: Filepath to set of pretrained
          ICNN weights.
        clear_data: Whether to clear storage folder before running.
    """
    # pylint: disable=too-many-locals, too-many-arguments
    if run_name == "":
        run_name = f'run_{str(int(time.time()))}'

    print(f'\n\tCreating (or restarting) run: {run_name}' \
         + f'\n\ton system: {system}' \
         + f'\n\twith BundleSDF cycle iteration: {cycle_iteration}' \
         + f'\n\tusing ContactNets: {contactnets}' \
         + f'\n\tregenerate: {regenerate}' \
         + f'\n\tdataset_size: {dataset_size}' \
         + f'\n\twith pretrained ICNN weights at: ' \
         + f'{pretrained_icnn_weights_filepath}' \
         + f'\n\tclear_data: {clear_data}\n')
    
    # First step, clear out data on disk for a fresh start.
    data_asset = DATA_ASSETS[system]
    pose_source = 'tagslam_toss' if cycle_iteration == 0 else \
        f'bundlesdf_toss_iteration_{cycle_iteration}'
    # where to store data
    storage_name = file_utils.assure_created(
        os.path.join(file_utils.RESULTS_DIR, data_asset, pose_source)
    )

    if clear_data:
        os.system(f'rm -r {file_utils.storage_dir(storage_name)}')

    # Next, build the configuration of the learning experiment.

    # Describes the optimizer settings; by default, the optimizer is Adam.
    optimizer_config = OptimizerConfig(lr=Float(LRS[system]),
                                       wd=Float(WDS[system]),
                                       patience=PATIENCE,
                                       epochs=EPOCHS,
                                       batch_size=Int(int(dataset_size/2)))

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

    # how to slice trajectories into training datapoints
    slice_config = TrajectorySliceConfig(
        t_prediction=1 if contactnets else T_PREDICTION)

    # Describes configuration of the data.
    data_config = DataConfig(dt=DT,
                             train_fraction=0.7,
                             valid_fraction=0.3,
                             test_fraction=0.0,
                             slice_config=slice_config,
                             update_dynamically=False,
                             dataset_size=dataset_size)

    # Combines everything into config for entire experiment.
    experiment_config = DrakeMultibodyLearnableExperimentConfig(
        storage=storage_name,
        run_name=run_name,
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
    experiment = DrakeMultibodyLearnableExperiment(experiment_config)

    # Prepare data.
    print('Preparing data.')
    # Specify directory with [T, n_x] tensor files saved as 0.pt, 1.pt, ...
    # See :mod:`dair_pll.state_space` for state format.
    import_directory = file_utils.get_asset(
        os.path.join(data_asset, pose_source)
    )
    file_utils.import_data_to_storage(storage_name,
                                      import_data_dir=import_directory,
                                      num=dataset_size)
        
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
@click.option('--system',
              type=click.Choice(SYSTEMS, case_sensitive=True),
              default=CUBE_SYSTEM)
@click.option('--cycle-iteration',
              type=int,
              default=1,
              help="BundleSDF iteration number (0 means use TagSLAM poses).")
@click.option('--contactnets/--prediction',
              default=True,
              help="whether to train/test with ContactNets/prediction loss.")
@click.option('--regenerate/--no-regenerate',
              default=False,
              help="whether save updated URDF's each epoch.")
@click.option('--dataset-size',
              type=int,
              default=512,
              help="dataset size")
@click.option('--pretrained',
              type=str,
              default=None,
              help='pretrained weights of Homonogeneous ICNN')
@click.option('--clear-data/--keep-data',
              default=False,
              help="Whether to clear storage folder before running.")

def main_command(run_name: str, system: str, cycle_iteration: int,
                 contactnets: bool, regenerate: bool, dataset_size: int,
                 pretrained: str, clear_data: bool):
    # pylint: disable=too-many-arguments
    """Executes main function with argument interface."""
    main(run_name, system, cycle_iteration, contactnets, regenerate,
         dataset_size, pretrained_icnn_weights_filepath=pretrained,
         clear_data=clear_data)



if __name__ == '__main__':
    main_command()  # pylint: disable=no-value-for-parameter
