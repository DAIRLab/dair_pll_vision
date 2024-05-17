"""Simple script to restart an experiment that has already started."""

import click
import os
import os.path as op
from typing import cast, Tuple

from dair_pll import file_utils
from dair_pll.experiment import default_epoch_callback
from dair_pll.multibody_learnable_system import MultibodyLearnableSystem
from dair_pll.vision_config import VisionExperiment, VisionExperimentConfig, \
    VISION_SYSTEMS, VISION_CUBE_SYSTEM



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
         cycle_iteration: int = 1):
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

    # Makes experiment.
    experiment = VisionExperiment(experiment_config)

    # Trains system and saves final results.
    print(f'\nTraining the model.')
    learned_system, _stats = experiment.generate_results(default_epoch_callback)

    # Save the final urdf.
    print(f'\nSaving the final learned URDF.')
    learned_system = cast(MultibodyLearnableSystem, learned_system)
    learned_system.generate_updated_urdfs('best')
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
def main_command(run_name: str, vision_asset: str, cycle_iteration: int):
    """Executes main function with argument interface."""
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
    
    pll_run_id = run_name
    if not pll_run_id.startswith('pll_id_'):
        pll_run_id = f'pll_id_{pll_run_id}'

    main(pll_run_id, system, start_toss, end_toss, cycle_iteration)


if __name__ == '__main__':
    main_command()  # pylint: disable=no-value-for-parameter
