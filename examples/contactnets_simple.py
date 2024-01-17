"""Simple ContactNets/differentiable physics learning examples."""
# pylint: disable=E1103
import os
import time
from typing import cast

import click
import numpy as np
import torch
from torch import Tensor

from dair_pll import file_utils
from dair_pll.dataset_generation import DataGenerationConfig, \
    ExperimentDatasetGenerator
from dair_pll.data_config import TrajectorySliceConfig, DataConfig
from dair_pll.drake_experiment import \
    DrakeMultibodyLearnableExperiment, DrakeSystemConfig, \
    MultibodyLearnableSystemConfig, MultibodyLosses, \
    DrakeMultibodyLearnableExperimentConfig
from dair_pll.experiment import default_epoch_callback
from dair_pll.experiment_config import OptimizerConfig
from dair_pll.hyperparameter import Float, Int
from dair_pll.multibody_learnable_system import MultibodyLearnableSystem
from dair_pll.state_space import UniformSampler, GaussianWhiteNoiser
from dair_pll.system import System

CUBE_SYSTEM = 'cube'
ELBOW_SYSTEM = 'elbow'
BUNDLESDF_CUBE_SYSTEM = 'bundlesdf_cube'
BUNDLESDF_PRISM_SYSTEM = 'bundlesdf_prism'
BUNDLESDF_TOBLERONE_SYSTEM = 'bundlesdf_toblerone'
SYSTEMS = [CUBE_SYSTEM, ELBOW_SYSTEM,
           BUNDLESDF_CUBE_SYSTEM,
           BUNDLESDF_PRISM_SYSTEM,
           BUNDLESDF_TOBLERONE_SYSTEM]
SIM_SOURCE = 'simulation'
REAL_SOURCE = 'real'
DYNAMIC_SOURCE = 'dynamic'
DATA_SOURCES = [SIM_SOURCE, REAL_SOURCE, DYNAMIC_SOURCE]

# File management.
CUBE_DATA_ASSET = 'contactnets_cube'
ELBOW_DATA_ASSET = 'contactnets_elbow'
BUNDLESDF_CUBE_DATA_ASSET = 'bundlesdf_cube'
PRISM_DATA_ASSET = 'bundlesdf_prism'
TOBLERONE_DATA_ASSET = 'bundlesdf_toblerone'
CUBE_BOX_URDF_ASSET = 'contactnets_cube.urdf'
CUBE_MESH_URDF_ASSET = 'contactnets_cube_mesh.urdf'
ELBOW_BOX_URDF_ASSET = 'contactnets_elbow.urdf'
ELBOW_MESH_URDF_ASSET = 'contactnets_elbow_mesh.urdf'
BUNDLESDF_MESH_URDF_ASSET = 'bundlesdf_cube_mesh.urdf'
PRISM_MESH_URDF_ASSET = 'bundlesdf_prism_mesh.urdf'
TOBLERONE_MESH_URDF_ASSET = 'bundlesdf_toblerone_mesh.urdf'

DATA_ASSETS = {CUBE_SYSTEM: CUBE_DATA_ASSET,
               ELBOW_SYSTEM: ELBOW_DATA_ASSET,
               BUNDLESDF_CUBE_SYSTEM: BUNDLESDF_CUBE_DATA_ASSET,
               BUNDLESDF_PRISM_SYSTEM: PRISM_DATA_ASSET,
               BUNDLESDF_TOBLERONE_SYSTEM: TOBLERONE_DATA_ASSET}

MESH_TYPE = 'mesh'
BOX_TYPE = 'box'
CUBE_URDFS = {MESH_TYPE: CUBE_MESH_URDF_ASSET, BOX_TYPE: CUBE_BOX_URDF_ASSET}
ELBOW_URDFS = {MESH_TYPE: ELBOW_MESH_URDF_ASSET, BOX_TYPE: ELBOW_BOX_URDF_ASSET}
BUNDLESDF_CUBE_URDFS = {MESH_TYPE: BUNDLESDF_MESH_URDF_ASSET}
PRISM_URDFS = {MESH_TYPE: PRISM_MESH_URDF_ASSET}
TOBLERONE_URDFS = {MESH_TYPE: TOBLERONE_MESH_URDF_ASSET}
URDFS = {CUBE_SYSTEM: CUBE_URDFS,
         ELBOW_SYSTEM: ELBOW_URDFS,
         BUNDLESDF_CUBE_SYSTEM: BUNDLESDF_CUBE_URDFS,
         BUNDLESDF_PRISM_SYSTEM: PRISM_URDFS,
         BUNDLESDF_TOBLERONE_SYSTEM: TOBLERONE_URDFS}


# Data configuration.
DT = 0.0333 #0.0068 # 1/frame rate of the camera

# Generation configuration.
CUBE_X_0 = torch.tensor([
    -0.525, 0.394, -0.296, -0.678, 0.186, 0.026, 0.222, 1.463, -4.854, 9.870,
    0.014, 1.291, -0.212
])
ELBOW_X_0 = torch.tensor(
    [1., 0., 0., 0., 0., 0., 0.21 + .015, np.pi, 0., 0., 0., 0., 0., -.075, 0.])
# TODO: seems not using this for real experiments
BOTTLE_X_0 = torch.tensor(
    [1., 0., 0., 0., 0., 0., 0.21 + .015, 0., 0., 0., 0., 0., -.075])
NAPKIN_X_0 = torch.tensor(
    [1., 0., 0., 0., 0., 0., 0.21 + .015, 0., 0., 0., 0., 0., -.075])
PRISM_X_0 = torch.tensor(
    [1., 0., 0., 0., 0., 0., 0.21 + .015, 0., 0., 0., 0., 0., -.075])
TOBLERONE_X_0 = torch.tensor(
    [1., 0., 0., 0., 0., 0., 0.21 + .015, 0., 0., 0., 0., 0., -.075])
X_0S = {CUBE_SYSTEM: CUBE_X_0, 
        ELBOW_SYSTEM: ELBOW_X_0,
        BUNDLESDF_CUBE_SYSTEM: CUBE_X_0,
        BUNDLESDF_PRISM_SYSTEM: PRISM_X_0,
        BUNDLESDF_TOBLERONE_SYSTEM: TOBLERONE_X_0}
CUBE_SAMPLER_RANGE = 0.1 * torch.ones(CUBE_X_0.nelement() - 1)
ELBOW_SAMPLER_RANGE = torch.tensor([
    2 * np.pi, 2 * np.pi, 2 * np.pi, .03, .03, .015, np.pi, 6., 6., 6., .5, .5,
    .075, 6.
])
SAMPLER_RANGES = {
    CUBE_SYSTEM: CUBE_SAMPLER_RANGE,
    ELBOW_SYSTEM: ELBOW_SAMPLER_RANGE
}
TRAJECTORY_LENGTHS = {CUBE_SYSTEM: 80, ELBOW_SYSTEM: 120}

# Training data configuration.
T_PREDICTION = 1

# Optimization configuration.
CUBE_LR = 1e-3
ELBOW_LR = 1e-3
PRISM_LR = 1e-3
TOBLERONE_LR = 1e-3
LRS = {CUBE_SYSTEM: CUBE_LR,
       ELBOW_SYSTEM: ELBOW_LR,
       BUNDLESDF_CUBE_SYSTEM: CUBE_LR,
       BUNDLESDF_PRISM_SYSTEM: PRISM_LR,
       BUNDLESDF_TOBLERONE_SYSTEM: TOBLERONE_LR}
CUBE_WD = 0.0
ELBOW_WD = 1e-4
PRISM_WD = 0.0
TOBLERONE_WD = 0.0
WDS = {CUBE_SYSTEM: CUBE_WD,
       ELBOW_SYSTEM: ELBOW_WD,
       BUNDLESDF_CUBE_SYSTEM: CUBE_WD,
       BUNDLESDF_PRISM_SYSTEM: PRISM_WD,
       BUNDLESDF_TOBLERONE_SYSTEM: TOBLERONE_WD}
EPOCHS = 100 #500
PATIENCE = EPOCHS

WANDB_PROJECT = 'dair_pll-examples'


def main(run_name: str = "",
         system: str = CUBE_SYSTEM,
         source: str = SIM_SOURCE,
         contactnets: bool = True,
         box: bool = True,
         regenerate: bool = False,
         dataset_size: int = 512,
         pretrained_icnn_weights_filepath: str = None,
         clear_data: bool = False):
    """Execute ContactNets basic example on a system.

    Args:
        run_name: name of experiment run.
        system: Which system to learn.
        source: Where to get data from.
        contactnets: Whether to use ContactNets or prediction loss
        box: Whether to represent geometry as box or mesh.
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
         + f'\n\ton system: {system} \n\twith source: {source}' \
         + f'\n\tusing ContactNets: {contactnets}' \
         + f'\n\tregenerate: {regenerate}' \
         + f'\n\tdataset_size: {dataset_size}' \
         + f'\n\twith pretrained ICNN weights at: {pretrained_icnn_weights_filepath}' \
         + f'\n\tclear_data: {clear_data}\n')

    # First step, clear out data on disk for a fresh start.
    simulation = source == SIM_SOURCE
    dynamic = source == DYNAMIC_SOURCE
    data_asset = DATA_ASSETS[system]
    # where to store data
    storage_name = file_utils.assure_created(
        os.path.join(file_utils.RESULTS_DIR, data_asset)
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
    # This is a configuration for a DrakeSystem, which wraps a Drake
    # simulation for the described URDFs.
    # first, select urdfs
    urdf_asset = URDFS[system][BOX_TYPE if box else MESH_TYPE]
    urdf = file_utils.get_asset(urdf_asset)
    urdfs = {system: urdf}
    base_config = DrakeSystemConfig(urdfs=urdfs)

    # Describes the learnable system. The MultibodyLearnableSystem type
    # learns a multibody system, which is initialized as the system in the
    # given URDFs.
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

    # Describes configuration of the data
    data_config = DataConfig(dt=DT,
                             train_fraction=1.0 if dynamic else 0.7,
                             valid_fraction=0.0 if dynamic else 0.3,
                             test_fraction=0.0 if dynamic else 0.0,
                             slice_config=slice_config,
                             update_dynamically=dynamic,
                             dataset_size=dataset_size)

    # Combines everything into config for entire experiment.
    experiment_config = DrakeMultibodyLearnableExperimentConfig(
        storage=storage_name,
        run_name=run_name,
        base_config=base_config,
        learnable_config=learnable_config,
        optimizer_config=optimizer_config,
        data_config=data_config,
        full_evaluation_period=EPOCHS if dynamic else 1,
        visualize_learned_geometry=True,
        run_wandb=True,
        wandb_project=WANDB_PROJECT
    )

    # Makes experiment.
    print('Making experiment.')
    experiment = DrakeMultibodyLearnableExperiment(experiment_config)

    # Prepare data.
    print('Preparing data.')
    x_0 = X_0S[system]
    if simulation:

        # For simulation, specify the following:
        data_generation_config = DataGenerationConfig(
            dt=DT,
            # timestep.
            n_pop=dataset_size,
            # How many trajectories to simulate
            trajectory_length=TRAJECTORY_LENGTHS[system],
            # trajectory length
            x_0=x_0,
            # A nominal initial state
            sampler_type=UniformSampler,
            # use uniform distribution to sample ``x_0``
            sampler_ranges=SAMPLER_RANGES[system],
            # How much to vary initial states around ``x_0``
            noiser_type=GaussianWhiteNoiser,
            # Distribution of noise in trajectory data (Gaussian).
            static_noise=torch.zeros(x_0.nelement() - 1),
            # constant-in-time noise standard deviations (zero in this case)
            dynamic_noise=torch.zeros(x_0.nelement() - 1),
            # i.i.d.-in-time noise standard deviations (zero in this case)
            storage=storage_name
            # where to store trajectories
        )

        generator = ExperimentDatasetGenerator(experiment.get_base_system(),
                                               data_generation_config)
        generator.generate()

    else:
        # otherwise, specify directory with [T, n_x] tensor files saved as
        # 0.pt, 1.pt, ...
        # See :mod:`dair_pll.state_space` for state format.
        import_directory = file_utils.get_asset(data_asset)
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
    learned_system, stats = experiment.generate_results(
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
@click.option('--source',
              type=click.Choice(DATA_SOURCES, case_sensitive=True),
              default=SIM_SOURCE)
@click.option('--contactnets/--prediction',
              default=True,
              help="whether to train/test with ContactNets/prediction loss.")
@click.option('--box/--mesh',
              default=True,
              help="whether to represent geometry as box or mesh.")
@click.option('--regenerate/--no-regenerate',
              default=False,
              help="whether save updated URDF's each epoch.")
@click.option('--dataset-size',
              default=512,
              help="dataset size")
@click.option('--pretrained',
              type=str,
              default=None,
              help='pretrained weights of Homonogeneous ICNN')
@click.option('--clear-data/--keep-data',
              default=False,
              help="Whether to clear storage folder before running.")

def main_command(run_name: str, system: str, source: str, contactnets: bool,
                 box: bool, regenerate: bool, dataset_size: int,
                 pretrained: str, clear_data: bool):
    # pylint: disable=too-many-arguments
    """Executes main function with argument interface."""
    if system == ELBOW_SYSTEM and source == REAL_SOURCE:
        raise NotImplementedError('Elbow real-world data not supported!')
    main(run_name, system, source, contactnets, box, regenerate, dataset_size,
         pretrained_icnn_weights_filepath=pretrained, clear_data=clear_data)



if __name__ == '__main__':
    main_command()  # pylint: disable=no-value-for-parameter
