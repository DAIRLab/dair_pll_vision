"""Utility functions for managing saved files for training models.

File system is organized around a "storage" directory associated with data and
training runs. The functions herein can be used to return absolute paths of and
summary information about the content of this directory.
"""
import glob
import json
import os
import numpy as np
import pickle
import random
from copy import deepcopy
from os import path
from typing import List, Callable, BinaryIO, Any, TextIO, Optional, Tuple, \
    Union, Dict

import torch
from torch import Tensor

from dair_pll.experiment_config import SupervisedLearningExperimentConfig

TRAJ_EXTENSION = '.pt'  # trajectory file
HYPERPARAMETERS_EXTENSION = '.json'  # hyperparameter set file
STATS_EXTENSION = '.pkl'  # experiment statistics
CONFIG_EXTENSION = '.pkl'
HUMAN_READABLE_CONFIG_EXTENSION = '.json'
CHECKPOINT_EXTENSION = '.pt'
DATA_SUBFOLDER_NAME = 'data'
LEARNING_DATA_SUBFOLDER_NAME = 'learning'
GROUND_TRUTH_DATA_SUBFOLDER_NAME = 'ground_truth'
RUNS_SUBFOLDER_NAME = '.'  #'runs'
STUDIES_SUBFOLDER_NAME = 'studies'
URDFS_SUBFOLDER_NAME = 'urdfs'
WANDB_SUBFOLDER_NAME = 'wandb'
GEOM_FOR_BSDF_SUBFOLDER_NAME = 'geom_for_bsdf'
GEOM_FOR_PLL_SUBFOLDER_NAME = 'geom_for_pll'
BSDF_FROM_SUPPORT_SUBSUBFOLDER_NAME = 'from_support_points'
BSDF_FROM_MESH_SUBSUBFOLDER_NAME = 'from_mesh_surface'
BSDF_SUPPORT_DIRECTIONS_NAME = 'support_directions.pt'
BSDF_SUPPORT_POINTS_NAME = 'support_points.pt'
BSDF_SUPPORT_SCALARS_NAME = 'support_scalars.pt'
BSDF_MESH_FILENAME = 'mesh.obj'
BSDF_SUBFOLDER_NAME = 'geom_for_bsdf'
EXPORT_POINTS_DEFAULT_NAME = 'support_points.pt'
EXPORT_DIRECTIONS_DEFAULT_NAME = 'support_directions.pt'
EXPORT_FORCES_DEFAULT_NAME = 'support_point_normal_forces.pt'
EXPORT_STATES_DEFAULT_NAME = 'support_point_states.pt'
EXPORT_TOSS_FRAME_IDX_DEFAULT_NAME = 'tosses_and_frames.pt'
EXPORT_POINTS_WITH_SDF_DEFAULT_NAME = 'ps.pt'
EXPORT_SDFS_FOR_POINTS_DEFAULT_NAME = 'sdfs.pt'
EXPORT_POINTS_WITH_SDF_BOUND_DEFAULT_NAME = 'vs.pt'
EXPORT_SDF_BOUNDS_FOR_POINTS_DEFAULT_NAME = 'sdf_bounds.pt'
EXPORT_POINTS_WITH_SDF_GRADIENT_DEFAULT_NAME = 'ws.pt'
EXPORT_SDF_GRADIENTS_FOR_POINTS_DEFAULT_NAME = 'w_normals.pt'
TRAJECTORY_GIF_DEFAULT_NAME = 'trajectory.gif'
FINAL_EVALUATION_NAME = f'statistics{STATS_EXTENSION}'
HYPERPARAMETERS_FILENAME = f'optimal_hyperparameters{HYPERPARAMETERS_EXTENSION}'
CONFIG_FILENAME = f'config{CONFIG_EXTENSION}'
CHECKPOINT_FILENAME = f'checkpoint{CHECKPOINT_EXTENSION}'
HUMAN_READABLE_CONFIG_FILENAME = f'config{HUMAN_READABLE_CONFIG_EXTENSION}'
"""str: extensions for saved files"""


def assure_created(directory: str) -> str:
    """Wrapper to put around directory paths which ensure their existence.

    Args:
        directory: Path of directory that may not exist.

    Returns:
        ``directory``, Which is ensured to exist by recursive mkdir.
    """
    directory = path.abspath(directory)
    if not path.exists(directory):
        assure_created(path.dirname(directory))
        os.mkdir(directory)
    return directory


MAIN_DIR = path.dirname(path.dirname(__file__))
ASSETS_DIR = assure_created(os.path.join(MAIN_DIR, 'assets'))
RESULTS_DIR = assure_created(os.path.join(MAIN_DIR, 'results'))
# str: locations of key static directories


def get_asset(asset_file_basename: str) -> str:
    """Gets

    Args:
        asset_file_basename: Basename of asset file located in ``ASSET_DIR``

    Returns:
        Asset's absolute path.
    """
    return os.path.join(ASSETS_DIR, asset_file_basename)


def assure_storage_tree_created(storage_name: str) -> None:
    """Assure that all subdirectories of specified storage are created.

    Args:
        storage_name: name of storage directory.
    """
    storage_directories = [data_dir, all_runs_dir,
                           all_studies_dir]  # type: List[Callable[[str],str]]

    for directory in storage_directories:
        assure_created(directory(storage_name))


def get_run_indices_in_dir(directory: str,
                           extension: str = TRAJ_EXTENSION) -> List[int]:
    """Return list of run indices in storage directory.

    Args:
        directory: Directory in which to look for runs.
        extension: Extension of files to be included.

    Returns:
        List of run indices.
    """
    run_names = [path.basename(file) for file in 
                 glob.glob(path.join(directory, './[0-9]*' + extension))]
    run_numbers = [int(name.split('.')[0]) for name in run_names]
    return run_numbers


def import_data_to_storage(storage_name: str, import_data_dir: str,
                           num: int = None) -> None:
    """Import data in external folder into data directory.

    Args:
        storage_name: Name of storage for data import.
        import_data_dir: Directory to import data from.
        num: Number of trajectories to import.  If larger than the number of
            trajectories present in import_data_dir, will import them all.
    """
    output_directories = [
        ground_truth_data_dir(storage_name),
        learning_data_dir(storage_name)
    ]
    data_traj_count = get_numeric_file_count(import_data_dir, TRAJ_EXTENSION)
    run_indices = get_run_indices_in_dir(import_data_dir, TRAJ_EXTENSION)

    target_traj_number = data_traj_count if num is None else \
                         min(num, data_traj_count)

    # Check if data is synchronized already.
    for output_directory in output_directories:
        storage_traj_count = get_numeric_file_count(output_directory,
                                                    TRAJ_EXTENSION)

        # Add trajectories if more requested than already present in output dir.
        if storage_traj_count != target_traj_number:

            # If output directory is empty and all trajectories are desired,
            # copy them all over, numbering from 0.
            if (storage_traj_count == 0) and \
               (target_traj_number >= data_traj_count):
                for output_dir in output_directories:
                    os.system(f'rm -r {output_dir}')
                    os.system(f'cp -r {import_data_dir} {output_dir}')
                    reorder_trajectories(output_dir, TRAJ_EXTENSION)

            # If output directory is empty and a subset is desired, copy a
            # random subset of trajectories.
            elif (storage_traj_count == 0) and \
                 (target_traj_number < data_traj_count):
                random.shuffle(run_indices)
                run_indices = run_indices[:target_traj_number]
                for output_dir in output_directories:
                    os.system(f'rm -r {output_dir}')
                    os.system(f'mkdir {output_dir}')

                    # Copy over a random selection of trajectories, numbering
                    # from 0.
                    for i, run in zip(range(target_traj_number), run_indices):
                        os.system(f'cp {import_data_dir}/{run}.pt ' + \
                                  f'{output_dir}/{i}.pt')

            # If output directory has fewer trajectories than desired, add a
            # random subset of additional trajectories.
            elif (storage_traj_count < target_traj_number):
                n_to_add = target_traj_number - storage_traj_count
                new_traj_idx = storage_traj_count

                random.shuffle(run_indices)
                traj_idx_to_try = 0

                # Need to check that we don't add a duplicate trajectory.
                while n_to_add > 0:
                    import_traj_name = f'{run_indices[traj_idx_to_try]}.pt'
                    import_traj = torch.load(path.join(import_data_dir,
                                                    import_traj_name))
                    
                    traj_idx_to_try += 1

                    if check_duplicate_trajectory(output_directory, import_traj):
                        continue

                    for output_dir in output_directories:
                        os.system(
                            f'cp {import_data_dir}/{import_traj_name} ' + \
                            f'{output_dir}/{new_traj_idx}.pt')

                    n_to_add -= 1
                    new_traj_idx += 1


            # If output directory has the right number or more of trajectories
            # than desired, all good and can continue.
            else:
                continue

            # Can terminate outer loop over output directories since all output
            # directories are written to at once.
            return
        

def reorder_trajectories(directory: str, extension: str = TRAJ_EXTENSION
                         ) -> None:
    """Rename trajectories in directory to be numbered from 0.

    Args:
        directory: Directory to reorder trajectories in.
        extension: Extension of files to be reordered.
    """
    traj_files = glob.glob(path.join(directory, f'*{extension}'))
    for i, traj_file in enumerate(traj_files):
        os.rename(traj_file, path.join(directory, f'{i}{extension}'))
        

def check_duplicate_trajectory(data_dir: str, traj: Tensor) -> bool:
    """Checks if a trajectory is already represented in a data directory."""
    existing_traj_files = glob.glob(
        path.join(data_dir, './[0-9]*' + TRAJ_EXTENSION)
    )
    for existing_traj_file in existing_traj_files:
        existing_traj = torch.load(existing_traj_file)
        if torch.all(traj == existing_traj).item():
            return True
    return False


def storage_dir(storage_name: str, create: bool = True) -> str:
    """Absolute path of storage directory"""
    if create:
        return assure_created(os.path.join(RESULTS_DIR, storage_name))
    else:
        return path.join(RESULTS_DIR, storage_name)


def data_dir(storage_name: str) -> str:
    """Absolute path of data folder."""
    return assure_created(
        path.join(storage_dir(storage_name), DATA_SUBFOLDER_NAME))


def learning_data_dir(storage_name: str) -> str:
    """Absolute path of folder for data preprocessed for
    training/validation."""
    return assure_created(
        path.join(data_dir(storage_name), LEARNING_DATA_SUBFOLDER_NAME))


def ground_truth_data_dir(storage_name: str) -> str:
    """Absolute path of folder for raw unprocessed trajectories."""
    return assure_created(
        path.join(data_dir(storage_name), GROUND_TRUTH_DATA_SUBFOLDER_NAME))


def all_runs_dir(storage_name: str, create: bool = True) -> str:
    """Absolute path of tensorboard storage folder"""
    if create:
        return assure_created(
            path.join(storage_dir(storage_name), RUNS_SUBFOLDER_NAME))
    else:
        return path.join(storage_dir(storage_name, create=False),
                         RUNS_SUBFOLDER_NAME)


def all_studies_dir(storage_name: str) -> str:
    """Absolute path of tensorboard storage folder"""
    return assure_created(
        path.join(storage_dir(storage_name), STUDIES_SUBFOLDER_NAME))


def delete(file_name: str) -> None:
    """Removes file at path specified by ``file_name``"""
    if path.exists(file_name):
        os.remove(file_name)


def get_numeric_file_count(directory: str,
                           extension: str = TRAJ_EXTENSION) -> int:
    """Count number of whole-number-named files.

    If folder ``/fldr`` has contents (7.pt, 11.pt, 4.pt), then::

        get_numeric_file_count("/fldr", ".pt") == 3

    Args:
        directory: Directory to tally file count in
        extension: Extension of files to be counted

    Returns:
        Number of files in specified ``directory`` with specified
        ``extension`` that have an integer basename.
    """
    return len(glob.glob(path.join(directory, './[0-9]*' + extension)))


def get_trajectory_count(directory: str) -> int:
    """Cound number of trajectories on disk in given directory."""
    return get_numeric_file_count(directory, TRAJ_EXTENSION)


def trajectory_file(trajectory_dir: str, num_trajectory: int) -> str:
    """Absolute path of specific trajectory in storage"""
    return path.join(trajectory_dir,
                     f'{int(num_trajectory)}{TRAJ_EXTENSION}')


def run_dir(storage_name: str, run_name: str, create: bool = True) -> str:
    """Absolute path of run-specific storage folder."""
    if create:
        return assure_created(path.join(all_runs_dir(storage_name), run_name))
    else:
        return path.join(all_runs_dir(storage_name, create=False), run_name)


def get_trajectory_video_filename(storage_name: str, run_name: str) -> str:
    """Return the filepath of the temporary rollout video gif."""
    return path.join(run_dir(storage_name, run_name),
                     TRAJECTORY_GIF_DEFAULT_NAME)


def get_learned_urdf_dir(storage_name: str, run_name: str) -> str:
    """Absolute path of learned model URDF storage directory."""
    return assure_created(
        path.join(run_dir(storage_name, run_name), URDFS_SUBFOLDER_NAME))


def wandb_dir(storage_name: str, run_name: str) -> str:
    """Absolute path of W&B storage folder"""
    return assure_created(
        path.join(run_dir(storage_name, run_name), WANDB_SUBFOLDER_NAME))


def geom_for_pll_dir(asset_subdirs: str, bundlesdf_id: str, iteration: int,
                     check_exists: bool = True) -> str:
    """Absolute path of geometry for PLL (a PLL input) storage folder"""
    geom_input_dir = path.join(
        get_asset(asset_subdirs), GEOM_FOR_PLL_SUBFOLDER_NAME,
        f'bundlesdf_iteration_{iteration}', bundlesdf_id
    )
    if check_exists:
        assert path.exists(geom_input_dir), f'No BundleSDF geometry input ' + \
            f'found at {geom_input_dir}'
    return geom_input_dir


def get_bundlesdf_geometry_data(asset_subdirs: str, bundlesdf_id: str,
                                iteration: int) -> Tuple[Tensor, Tensor]:
    """Load geometry data from BundleSDF."""
    geom_input_dir = geom_for_pll_dir(asset_subdirs, bundlesdf_id, iteration)
    assert BSDF_SUPPORT_DIRECTIONS_NAME in os.listdir(geom_input_dir) and \
        BSDF_SUPPORT_POINTS_NAME in os.listdir(geom_input_dir), \
        f'BundleSDF geometry input at {geom_input_dir} is incomplete; ' + \
        f'missing {BSDF_SUPPORT_DIRECTIONS_NAME} or {BSDF_SUPPORT_POINTS_NAME}.'

    bsdf_dirs = torch.load(path.join(geom_input_dir,
                                     BSDF_SUPPORT_DIRECTIONS_NAME))
    bsdf_pts = torch.load(path.join(geom_input_dir, BSDF_SUPPORT_POINTS_NAME))
    bsdf_ds = torch.load(path.join(geom_input_dir, BSDF_SUPPORT_SCALARS_NAME))

    assert bsdf_dirs.shape == bsdf_pts.shape, f'Expected BundleSDF support ' + \
        f'directions and points to have same shape, got {bsdf_dirs.shape} ' + \
        f'and {bsdf_pts.shape} respectively.'
    assert bsdf_dirs.ndim == 2, f'Expected BundleSDF support directions ' + \
        f'and points to be (N, 3) but got shape {bsdf_dirs.shape}.'
    assert bsdf_ds.ndim == 1, f'Expected BundleSDF support scalars to be ' + \
        f'(N,) but got shape {bsdf_ds.shape}.'
    assert bsdf_ds.shape[0] == bsdf_dirs.shape[0], f'Expected BundleSDF ' + \
        f'support directions/scalars/points to have same length, got ' + \
        f'{bsdf_dirs.shape[0]=}, {bsdf_ds.shape[0]=}, {bsdf_pts.shape[0]=}.'
    assert bsdf_dirs.shape[1] == 3, f'Expected BundleSDF support ' + \
        f'directions and points to be (N, 3) but got shape {bsdf_dirs.shape}.'

    return bsdf_dirs, bsdf_pts, bsdf_ds


def get_mesh_from_bundlesdf(asset_subdirs: str, bundlesdf_id: str,
                            iteration: int) -> str:
    """Load mesh data from BundleSDF."""
    geom_input_dir = geom_for_pll_dir(asset_subdirs, bundlesdf_id, iteration)
    assert BSDF_MESH_FILENAME in os.listdir(geom_input_dir), \
        f'BundleSDF geometry input at {geom_input_dir} is incomplete; ' + \
        f'missing {BSDF_MESH_FILENAME}.'

    return os.path.join(geom_input_dir, BSDF_MESH_FILENAME)


def get_vision_urdf_template_path() -> str:
    """Return the path to the vision URDF template file."""
    return get_asset('vision_template.urdf')


def geom_for_bsdf_dir(storage_name: str, run_name: str) -> str:
    """Absolute path of geometry for BundleSDF (a PLL output) storage folder"""
    return assure_created(
        path.join(run_dir(storage_name, run_name), GEOM_FOR_BSDF_SUBFOLDER_NAME))


def store_geom_for_bsdf(storage_name: str, run_name: str, points: Tensor,
                        directions: Tensor, normal_forces: Tensor,
                        states: Tensor) -> None:
    """Store geometry-related data exports for BundleSDF."""
    output_dir = geom_for_bsdf_dir(storage_name, run_name)
    torch.save(points, path.join(output_dir, EXPORT_POINTS_DEFAULT_NAME))
    torch.save(directions, path.join(output_dir, EXPORT_DIRECTIONS_DEFAULT_NAME))
    torch.save(normal_forces, path.join(output_dir, EXPORT_FORCES_DEFAULT_NAME))
    torch.save(states, path.join(output_dir, EXPORT_STATES_DEFAULT_NAME))


def store_sdf_for_bsdf(storage_name: str, run_name: str,
                       from_support_not_mesh: bool, ps: Tensor, sdfs: Tensor,
                       vs: Tensor, sdf_bounds: Tensor, ws: Tensor = None,
                       w_normals: Tensor = None) -> None:
    """Store SDF training data for BundleSDF."""
    bsdf_output_dir = geom_for_bsdf_dir(storage_name, run_name)
    subfolder = BSDF_FROM_SUPPORT_SUBSUBFOLDER_NAME if from_support_not_mesh \
        else BSDF_FROM_MESH_SUBSUBFOLDER_NAME
    output_dir = assure_created(path.join(bsdf_output_dir, subfolder))

    torch.save(ps, path.join(output_dir, EXPORT_POINTS_WITH_SDF_DEFAULT_NAME))
    torch.save(sdfs, path.join(output_dir, EXPORT_SDFS_FOR_POINTS_DEFAULT_NAME))
    torch.save(vs, path.join(output_dir,
                             EXPORT_POINTS_WITH_SDF_BOUND_DEFAULT_NAME))
    torch.save(sdf_bounds, path.join(output_dir,
                                     EXPORT_SDF_BOUNDS_FOR_POINTS_DEFAULT_NAME))
    
    if ws is not None and w_normals is not None:
        torch.save(ws,
                   path.join(output_dir,
                             EXPORT_POINTS_WITH_SDF_GRADIENT_DEFAULT_NAME))
        torch.save(w_normals,
                   path.join(output_dir,
                             EXPORT_SDF_GRADIENTS_FOR_POINTS_DEFAULT_NAME))


def get_trajectory_assets_from_config(storage_name: str, run_name: str) -> str:
    """NOTE: This only works for vision experiments since
    `full_asset_directory_path` is an attribute in VisionDataConfig."""
    config = load_configuration(storage_name, run_name)
    traj_dir = config.data_config.full_asset_directory_path

    toss_trajs = {}

    for file in os.listdir(traj_dir):
        if file.endswith('.pt'):
            toss = int(file.split('.')[0])
            traj = torch.load(path.join(traj_dir, file))
            toss_trajs[toss] = traj

    return toss_trajs


def get_evaluation_filename(storage_name: str, run_name: str) -> str:
    """Absolute path of experiment run statistics file."""
    return path.join(run_dir(storage_name, run_name), FINAL_EVALUATION_NAME)


def get_configuration_filename(storage_name: str, run_name: str,
                               human_readable: bool = False) -> str:
    """Absolute path of experiment configuration."""
    if human_readable:
        return path.join(run_dir(storage_name, run_name),
                         HUMAN_READABLE_CONFIG_FILENAME)
    return path.join(run_dir(storage_name, run_name), CONFIG_FILENAME)


def get_model_filename(storage_name: str, run_name: str) -> str:
    """Absolute path of experiment configuration."""
    return path.join(run_dir(storage_name, run_name), CHECKPOINT_FILENAME)


def study_dir(storage_name: str, study_name: str) -> str:
    """Absolute path of study-specific storage folder."""
    return assure_created(path.join(all_studies_dir(storage_name), study_name))


def hyperparameter_opt_run_name(study_name: str, trial_number: int) -> str:
    """Experiment run name for hyperparameter optimization trial."""
    return f'{study_name}_hyperparameter_opt_{trial_number}'

  
def sweep_run_name(study_name: str, sweep_run: int, n_train: int) -> str:
    """Experiment run name for dataset size sweep study."""
    return f'{study_name}_sweep_{sweep_run}_n_train_{n_train}'


def get_hyperparameter_filename(storage_name: str, study_name: str) -> str:
    """Absolute path of optimized hyperparameters for a study"""
    return path.join(study_dir(storage_name, study_name),
                     HYPERPARAMETERS_FILENAME)


def load_binary(filename: str, load_callback: Callable[[BinaryIO], Any]) -> Any:
    """Load binary file"""
    with open(filename, 'rb') as file:
        value = load_callback(file)
    return value


def load_string(filename: str,
                load_callback: Optional[Callable[[TextIO], Any]] = None) -> Any:
    """Load text file"""
    with open(filename, 'r', encoding='utf8') as file:
        if load_callback:
            value = load_callback(file)
        else:
            value = file.read()
    return value


def save_binary(filename: str, value: Any,
                save_callback: Callable[[Any, BinaryIO], None]) -> None:
    """Save binary file."""
    with open(filename, 'wb') as file:
        save_callback(value, file)


def save_string(
        filename: str,
        value: Any,
        save_callback: Optional[Callable[[Any, TextIO], None]] = None) -> None:
    """Save text file."""
    with open(filename, 'w', encoding='utf8') as file:
        if save_callback:
            save_callback(value, file)
        else:
            assert isinstance(value, str)
            file.write(value)


def load_configuration(storage_name: str, run_name: str,
                       do_type_check: bool = True) -> \
        Union[SupervisedLearningExperimentConfig, Any]:
    """Load configuration file."""
    configuration_filename = get_configuration_filename(storage_name, run_name)
    configuration = load_binary(configuration_filename, pickle.load)
    if do_type_check:
        assert isinstance(configuration, SupervisedLearningExperimentConfig), \
            f'Expected {SupervisedLearningExperimentConfig}, got ' \
            f'{type(configuration)}'
    return configuration


def save_configuration(storage_name: str, run_name: str,
                       config: SupervisedLearningExperimentConfig,
                       human_readable: bool = False) -> None:
    """Save configuration file."""
    configuration_filename = get_configuration_filename(storage_name, run_name)
    save_binary(configuration_filename, config, pickle.dump)

    if human_readable:
        configuration_filename = get_configuration_filename(
            storage_name, run_name, human_readable=True)
        serializable_config = make_serializable(config)

        save_string(configuration_filename,
                    json.dumps(serializable_config, indent=2))


def load_evaluation(storage_name: str, run_name: str) -> Any:
    """Load evaluation file."""
    evaluation_filename = get_evaluation_filename(storage_name, run_name)
    return load_binary(evaluation_filename, pickle.load)


def save_evaluation(storage_name: str, run_name: str, evaluation: Any) -> None:
    """Save evaluation file."""
    evaluation_filename = get_evaluation_filename(storage_name, run_name)
    save_binary(evaluation_filename, evaluation, pickle.dump)


def load_hyperparameters(storage_name: str, study_name: str) -> Any:
    """Load hyperparameter file."""
    hyperparameter_filename = get_hyperparameter_filename(
        storage_name, study_name)
    return load_string(hyperparameter_filename, json.load)


def save_hyperparameters(storage_name: str, study_name: str,
                         hyperparameters: Any) -> None:
    """Save hyperparameter file."""
    hyperparameter_filename = get_hyperparameter_filename(
        storage_name, study_name)
    save_string(hyperparameter_filename, hyperparameters, json.dump)


def make_serializable(obj: Any) -> Dict[str, Any]:
    """Converts a non-serializable object to a series of nested dictionaries
    with values of regular types so the result can be written to a json file,
    for example.  Recursively serializes subcomponents, skipping only cases
    when a deepest traced item cannot be serialized."""
    dict = deepcopy(obj.__dict__)
    for key, value in dict.items():
        try:
            json.dumps(value)
        except TypeError:
            try:
                dict[key] = make_serializable(value)
            except TypeError:
                dict[key] = '<UNSERIALIZABLE OBJECT>'
    return dict
