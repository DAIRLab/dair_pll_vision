
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, cast

import os.path as op
import re
import torch
from torch import Tensor

from dair_pll import file_utils
from dair_pll.data_config import DataConfig
from dair_pll.dataset_management import ExperimentDataManager, TrajectorySet
from dair_pll.drake_experiment import DrakeMultibodyLearnableExperimentConfig, \
    DrakeMultibodyLearnableExperiment
from dair_pll.experiment import LossCallbackCallable, TrainingState


VISION_CUBE_SYSTEM = 'vision_cube'
VISION_PRISM_SYSTEM = 'vision_prism'
VISION_TOBLERONE_SYSTEM = 'vision_toblerone'
VISION_MILK_SYSTEM = 'vision_milk'
VISION_SYSTEMS = ['vision_bottle', VISION_CUBE_SYSTEM, 'vision_egg',
                  'vision_half', VISION_MILK_SYSTEM, 'vision_napkin',
                  VISION_PRISM_SYSTEM, VISION_TOBLERONE_SYSTEM]


@dataclass
class VisionDataConfig(DataConfig):
    """Data configuration for vision experiments.  Requires timestep, fractions,
    and slice configuration from parent DataConfig, and additionally the below:
        - asset_subdirectories (str):  Asset subdirectory, e.g.
            vision_cube/cube_1.
        - tracker (str):  Tracker to use for pose estimation.  Can be tagslam or
            bundlesdf_iteration_X.
        - bundlesdf_id:  BundleSDF experiment ID for pose estimation.  This will
            be prepended with bundlesdf_id_ when searching for the assets.
    
    From the above information, the following attributes are set upon creation:
        - full_asset_directory_path (str):  Full path to the asset directory.
        - toss_name_regex (str):  Regular expression for toss filenames.
    """
    asset_subdirectories: str = None
    tracker: str = 'tagslam'
    bundlesdf_id: str = None
    full_asset_directory_path: str = None
    toss_name_regex: str = None

    def __post_init__(self):
        """Method to fill in dataset_size based on assets directory."""
        # The below attributes are to be automatically set upon creation.
        assert self.full_asset_directory_path is None
        assert self.toss_name_regex is None

        # Check the asset subdirectories match expectations.
        dirs = self.asset_subdirectories.split('/')
        self.asset_subdirectories = f'{dirs[0]}/{dirs[1]}'
        assert len(dirs) == 2, f'Expected system/dataset, got ' \
            f'{self.asset_subdirectories}'
        assert dirs[0] in VISION_SYSTEMS, f'Invalid system {dirs[0]}'

        assert dirs[0].split('_')[1] == dirs[1].split('_')[0], \
            f'Invalid/inconsistent system {dirs[0]} or dataset {dirs[1]}.'
        
        # Set the dataset size based on the number of tosses.
        tosses = dirs[1].split('_')[1]
        if bool(re.match(r'^\d+$', tosses)):
            multi = False
            first_toss, last_toss = int(tosses), int(tosses)
        elif bool(re.match(r'^\d+\d+$', tosses)):
            first_toss = int(tosses.split('-')[0])
            last_toss = int(tosses.split('-')[1])
            multi = True
        else:
            raise ValueError(f'Invalid dataset {dirs[1]}')
        self.dataset_size = last_toss - first_toss + 1

        # Check the tracker is among the possible options.
        self.full_asset_directory_path = file_utils.get_asset(
            op.join(self.asset_subdirectories, 'toss', self.tracker)
        )
        if self.tracker == 'tagslam':
            tagslam = True
            if not multi:
                self.full_asset_directory_path = file_utils.get_asset(
                    op.join(self.asset_subdirectories, 'toss')
                )
        else:
            tagslam = False
            assert bool(re.match(r'^bundlesdf_iteration_\d+$', self.tracker)), \
                f'Invalid tracker {self.tracker}.'
            assert self.bundlesdf_id is not None, f'Requires bundlesdf_id to ' \
                f'use tracker {self.tracker}'
            if multi:
                self.full_asset_directory_path = file_utils.get_asset(
                    op.join(self.asset_subdirectories, 'toss', self.tracker,
                            f'bundlesdf_id_{self.bundlesdf_id}')
                )
            
        assert op.isdir(self.full_asset_directory_path), f'No existing folder' \
            f' at {self.full_asset_directory_path}.'

        # Check that the expected files exist.
        if tagslam and not multi:
            assert op.exists(op.join(self.full_asset_directory_path,
                                     'tagslam.pt')), f'No tagslam.pt in ' \
                f'{self.full_asset_directory_path}'
            self.toss_name_regex = 'tagslam.pt'
        elif multi:
            for i in range(first_toss, last_toss+1):
                assert op.exists(
                    op.join(self.full_asset_directory_path, f'toss_{i}')
                ), f'Expected but did not find ' \
                    f'{self.full_asset_directory_path}/toss_{i}.pt.'
            self.toss_name_regex = r'^\d+.pt$'
        else:
            assert op.exists(
                op.join(self.full_asset_directory_path,
                        f'bundlesdf_id_{self.bundlesdf_id}.pt')
            ), f'Expected but did not find ' \
                f'{self.full_asset_directory_path}/' \
                f'bundlesdf_id_{self.bundlesdf_id}.pt.'
            self.toss_name_regex = f'bundlesdf_id_{self.bundlesdf_id}.pt'

        # Check validity of parameters.
        super().__post_init__()


@dataclass
class VisionExperimentConfig(DrakeMultibodyLearnableExperimentConfig):
    """Overwrites DrakeMultibodyLearnableExperimentConfig's data_config
    attribute to be of type VisionDataConfig.'"""
    data_config: VisionDataConfig = field(default_factory=VisionDataConfig)


class VisionExperimentDataManager(ExperimentDataManager):
    """Class for managing data for vision experiments.
    
    TODO:
        - figure out how to reload dataset for a resumed experiment
    """

    def __init__(
            self, config: VisionDataConfig,
            initial_split: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
    ) -> None:
        
        self.trajectory_dir = config.full_asset_directory_path
        self.config = config
        self.train_set = self.make_empty_trajectory_set()
        self.valid_set = self.make_empty_trajectory_set()
        self.test_set = self.make_empty_trajectory_set()
        self.n_sorted = 0
        self.n_target = torch.inf if config.dataset_size == 0 else \
                        config.dataset_size
        if initial_split:
            self.extend_trajectory_sets(initial_split)

    def extend_trajectory_sets(self, index_lists: Tuple[Tensor]) -> None:
        if self.config.dataset_size == 1:
            trajectory = torch.load(
                op.join(self.config.full_asset_directory_path,
                        self.config.toss_name_regex)
            )
            self.train_set.add_trajectories(
                [trajectory],
                Tensor([-1*int(self.config.asset_subdirectories.split('_')[-1])])
            )
            self.n_sorted = 1

        else:
            return super().extend_trajectory_sets(index_lists)
        
    def get_updated_trajectory_sets(
            self) -> Tuple[TrajectorySet, TrajectorySet, TrajectorySet]:
        """Returns an up-to-date partition of trajectories on disk, up to the
        target number of trajectories.

        Checks if some trajectories on disk have yet to be sorted,
        and supplements the (train, valid, test) sets with these additional
        trajectories before returning the updated sets.

        Returns:
            Training set.
            Validation set.
            Test set.
        """
        config = self.config
        n_any_on_disk = file_utils.get_trajectory_count(self.trajectory_dir,
                                                        numeric_only=False)
        n_num_on_disk = file_utils.get_trajectory_count(self.trajectory_dir)

        # There should only be non-number-named trajectories in the same folder
        # if the dataset size is 1.
        if n_any_on_disk != n_num_on_disk:
            assert self.config.dataset_size == 1, \
                f"Found non-numbered trajectories on disk, but dataset_size " \
                f"is greater than 1 ({self.config.dataset_size})."
            
            # Don't need to provide indices when dataset size is 1 because the
            # data_config specifies the trajectory name to access.
            self.extend_trajectory_sets(None)
            return self._trajectory_sets

        assert n_num_on_disk == self.config.dataset_size, \
            f"Dataset_size is {self.config.dataset_size} but" \
            f" only found {n_num_on_disk} trajectories on disk."
        
        if n_num_on_disk != self.n_sorted:
            assert self.n_sorted == 0, f"Expecting for vision experiments to " \
                f"sort all needed trajectories at once, but have " \
                f"{self.n_sorted} already sorted and {n_num_on_disk} on disk."
            
            traj_nums = file_utils.get_run_indices_in_dir(self.trajectory_dir)
            
            n_to_add = n_num_on_disk

            n_train = round(n_to_add * config.train_fraction)
            n_valid = round(n_to_add * config.valid_fraction)
            n_remaining = n_to_add - n_valid - n_train
            n_test = min(n_remaining, round(n_to_add * config.test_fraction))

            n_requested = n_train + n_valid + n_test
            assert n_requested == n_to_add

            trajectory_order = torch.randperm(n_to_add)[:n_to_add]
            train_indices = traj_nums[trajectory_order[:n_train]]
            trajectory_order = trajectory_order[n_train:]

            valid_indices = traj_nums[trajectory_order[:n_valid]]
            trajectory_order = trajectory_order[n_valid:]
            test_indices = traj_nums[trajectory_order[:n_test]]

            self.extend_trajectory_sets(
                (train_indices, valid_indices, test_indices))

        return self._trajectory_sets


class VisionExperiment(DrakeMultibodyLearnableExperiment):
    """Class for loading and training vision experiments.
    
    TODO:
        - load assets using VisionDataConfig
        - check that config.pkl contains VisionExperimentConfig things too
    """

    def __init__(self, config: VisionExperimentConfig) -> None:
        super().__init__(config)
        file_utils.save_configuration(config.storage, config.run_name, config,
                                      human_readable=True)

    def setup_learning_data_manager(self, checkpoint_filename: str
                                    ) -> VisionExperimentDataManager:
        """"""
        is_resumed = False
        training_state = None
        checkpoint_filename = file_utils.get_model_filename(
            self.config.storage, self.config.run_name)
        try:
            # if a checkpoint is saved from disk, attempt to load it.
            checkpoint_dict = torch.load(checkpoint_filename)
            training_state = TrainingState(**checkpoint_dict)
            print("Resumed from disk.")
            is_resumed = True
            self.learning_data_manager = VisionExperimentDataManager(
                self.config.data_config,
                training_state.trajectory_set_split_indices)
        except FileNotFoundError:
            self.learning_data_manager = VisionExperimentDataManager(
                self.config.data_config)
            
        return is_resumed, training_state
        