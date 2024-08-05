"""File with class extensions and variable definitions for BundleSDF/PLL vision
project."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, cast, Dict, Union, Any, Callable

import os.path as op
import pdb
import re
import time
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tensordict.tensordict import TensorDictBase

from dair_pll import file_utils
from dair_pll.data_config import DataConfig
from dair_pll.dataset_management import ExperimentDataManager, TrajectorySet
from dair_pll.drake_experiment import DrakeMultibodyLearnableExperimentConfig, \
    DrakeMultibodyLearnableExperiment, MultibodyLosses, \
    MultibodyLearnableSystemConfig
from dair_pll.experiment import LossCallbackCallable, TrainingState, \
    LOGGING_DURATION
from dair_pll.geometry import DeepSupportConvex
from dair_pll.system import System
from dair_pll.multibody_learnable_system import MultibodyLearnableSystem


VISION_CUBE_SYSTEM = 'vision_cube'
VISION_PRISM_SYSTEM = 'vision_prism'
VISION_TOBLERONE_SYSTEM = 'vision_toblerone'
VISION_MILK_SYSTEM = 'vision_milk'
VISION_SYSTEMS = ['vision_bottle', VISION_CUBE_SYSTEM, 'vision_egg',
                  'vision_half', VISION_MILK_SYSTEM, 'vision_napkin',
                  VISION_PRISM_SYSTEM, VISION_TOBLERONE_SYSTEM, 
                  'vision_bakingbox', 'vision_burger', 'vision_cardboard',
                  'vision_chocolate', 'vision_cream', 'vision_croc',
                  'vision_crushedcan', 'vision_duck', 'vision_gallon',
                  'vision_greencan', 'vision_hotdog', 'vision_icetray',
                  'vision_mug', 'vision_oatly', 'vision_pinkcan', 'vision_stapler',
                  'vision_styrofoam', 'vision_toothpaste',
                  'vision_robot_bakingbox_sticky_A']


@dataclass
class VisionDataConfig(DataConfig):
    """Data configuration for vision experiments.  Requires timestep, fractions,
    and slice configuration from parent DataConfig, and additionally the below:
        - asset_subdirectories (str):  Asset subdirectory, e.g.
            vision_cube/cube_1.
        - tracker (str):  Tracker to use for pose estimation.  Can be tagslam or
            bundlesdf_iteration_X.
        - bundlesdf_id:  BundleSDF experiment ID for pose estimation.  This will
            be ensured to have the bundlesdf_id_ prefix.
    
    From the above information, the following attributes are set upon creation:
        - full_asset_directory_path (str):  Full path to the asset directory,
            directly inside of which are the *.pt files to load.
    """
    asset_subdirectories: str = None
    tracker: str = 'tagslam'
    bundlesdf_id: str = None
    full_asset_directory_path: str = None
    dataset_size: int = None

    def __post_init__(self):
        """Method to fill in dataset_size based on assets directory."""
        # The below attributes are to be automatically set upon creation.
        assert self.full_asset_directory_path is None
        assert self.dataset_size is None

        # Make the BundleSDF ID have the expected prefix.
        if self.bundlesdf_id is not None:
            if not self.bundlesdf_id.startswith('bundlesdf_id_'):
                self.bundlesdf_id = f'bundlesdf_id_{self.bundlesdf_id}'

        # Check the asset subdirectories match expectations, e.g.
        # vision_cube/cube_2.
        dirs = self.asset_subdirectories.split('/')
        self.asset_subdirectories = f'{dirs[0]}/{dirs[1]}'
        assert len(dirs) == 2, f'Expected system/dataset, got ' \
            f'{self.asset_subdirectories}'
        assert dirs[0] in VISION_SYSTEMS, f'Invalid system {dirs[0]}'
        assert '_'.join(dirs[0].split('_')[1:]) == '_'.join(dirs[1].split('_')[:-1]), \
            f'Invalid/inconsistent system {dirs[0]} or dataset {dirs[1]}.'

        # Set the dataset size based on the number of tosses.
        tosses = dirs[1].split('_')[-1]
        if bool(re.match(r'^\d+$', tosses)):
            first_toss, last_toss = int(tosses), int(tosses)
        elif bool(re.match(r'^\d+-\d+$', tosses)):
            first_toss = int(tosses.split('-')[0])
            last_toss = int(tosses.split('-')[1])
        else:
            raise ValueError(f'Invalid dataset {dirs[1]}')
        self.dataset_size = last_toss - first_toss + 1

        # Check the tracker is among the possible options.
        tracking_dir = file_utils.get_asset(
            op.join(self.asset_subdirectories, 'toss', self.tracker)
        )
        if self.tracker == 'tagslam':
            self.full_asset_directory_path = tracking_dir
        else:
            assert bool(re.match(r'^bundlesdf_iteration_\d+$', self.tracker)), \
                f'Invalid tracker {self.tracker}.'
            assert self.bundlesdf_id is not None, f'Requires bundlesdf_id to ' \
                f'use tracker {self.tracker}'

            # Set the full asset directory path.
            self.full_asset_directory_path = file_utils.get_asset(
                op.join(tracking_dir, self.bundlesdf_id))

        assert op.isdir(self.full_asset_directory_path), f'No existing folder' \
            f' at {self.full_asset_directory_path}.'

        # Check that the expected files exist.
        assert file_utils.get_numeric_file_count(
            self.full_asset_directory_path) == self.dataset_size, f'Did not' + \
                f' find {self.dataset_size} trajectories in ' + \
                f'{self.full_asset_directory_path}.'

        # Check validity of parameters.
        super().__post_init__()


@dataclass
class VisionExperimentConfig(DrakeMultibodyLearnableExperimentConfig):
    """Overwrites DrakeMultibodyLearnableExperimentConfig's data_config
    attribute to be of type VisionDataConfig.'"""
    data_config: VisionDataConfig = field(default_factory=VisionDataConfig)


class VisionExperimentDataManager(ExperimentDataManager):
    """Class for managing data for vision experiments."""

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
        n_on_disk = file_utils.get_numeric_file_count(self.trajectory_dir)

        assert n_on_disk == self.config.dataset_size, \
            f"Dataset_size is {self.config.dataset_size} but" \
            f" only found {n_on_disk} trajectories on disk."
        
        if n_on_disk != self.n_sorted:
            assert self.n_sorted == 0, f"Expecting for vision experiments to " \
                f"sort all needed trajectories at once, but have " \
                f"{self.n_sorted} already sorted and {n_on_disk} on disk."
            
            traj_nums = Tensor(file_utils.get_run_indices_in_dir(
                self.trajectory_dir))
            
            n_to_add = n_on_disk

            n_train = round(n_to_add * config.train_fraction)
            n_valid = round(n_to_add * config.valid_fraction)
            n_remaining = n_to_add - n_valid - n_train
            n_test = min(n_remaining, round(n_to_add * config.test_fraction))

            n_requested = n_train + n_valid + n_test
            assert n_requested == n_to_add

            trajectory_order = torch.randperm(n_to_add)
            train_indices = traj_nums[trajectory_order[:n_train]]
            trajectory_order = trajectory_order[n_train:]

            valid_indices = traj_nums[trajectory_order[:n_valid]]
            trajectory_order = trajectory_order[n_valid:]
            test_indices = traj_nums[trajectory_order[:n_test]]

            self.extend_trajectory_sets(
                (train_indices, valid_indices, test_indices))

        return self._trajectory_sets


class VisionExperiment(DrakeMultibodyLearnableExperiment):
    """Class for loading and training vision experiments."""

    def __init__(self, config: VisionExperimentConfig) -> None:
        super().__init__(config)
        if config.learnable_config.loss == MultibodyLosses.VISION_LOSS:
            self.loss_callback = self.vision_loss
        file_utils.save_configuration(config.storage, config.run_name, config,
                                      human_readable=True)

        # Get precomputed functions in dictionary[string, callable] form.
        precomputed_functions = {}
        dirs = config.learnable_config.precomputed_function_directories
        if 'mass_matrix' in dirs.keys():
            precomputed_functions['mass_matrix'] = \
                get_precomputed_mass_matrix_function(dirs['mass_matrix'])
        if 'lagrangian_forces' in dirs.keys():
            precomputed_functions['lagrangian_forces'] = \
                get_precomputed_lagrangian_forces_function(
                    dirs['lagrangian_forces'])
        self.precomputed_functions = precomputed_functions

    def setup_learning_data_manager(self, checkpoint_filename: str
                                    ) -> VisionExperimentDataManager:
        """Sets up learning data manager for vision experiments."""
        is_resumed = False
        training_state = None
        checkpoint_filename = file_utils.get_model_filename(
            self.config.storage, self.config.run_name)
        try:
            # if a checkpoint is saved from disk, attempt to load it.
            checkpoint_dict = torch.load(checkpoint_filename, weights_only=True)
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

    def vision_loss(
            self, x_past: Tensor, x_future: Tensor, system: System,
            keep_batch: bool = False
    ) -> Tensor:
        """Loss function for vision experiments includes both ContactNets loss
        and an additional loss from BundleSDF's shape supervision."""
        if not isinstance(system, MultibodyLearnableSystem):
            print(f'Warning: vision loss requires a MultibodyLearnableSystem' +\
                  f' but got {type(system)} instead; skipping and returning 0.')
            return Tensor([0.0])

        contactnets_loss = self.contactnets_loss(
            x_past, x_future, system, keep_batch)
        bundlesdf_loss = self.bundlesdf_geometry_loss(system)

        # Combine the two terms.  The ContactNets loss is already scaled by the
        # appropriate weights, so only need to scale the BundleSDF loss.
        return contactnets_loss + system.w_bsdf * bundlesdf_loss

    def bundlesdf_geometry_loss(self, system: System) -> Tensor:
        """Loss function for matching vision-informed geometry estimate from
        BundleSDF."""
        # First check if this loss can be computed based on BundleSDF inputs.
        if self.config.data_config.bundlesdf_id is None or system.w_bsdf == 0.0:
            return Tensor([0.0])

        # First uncover the deep support convex network.
        all_geoms = system.multibody_terms.contact_terms.geometries
        deep_support_geom = None
        for geom in all_geoms:
            if type(geom) == DeepSupportConvex:
                assert deep_support_geom is None, f'Multiple ' + \
                    f'DeepSupportConvex found in {all_geoms}.'
                deep_support_geom = geom
        assert deep_support_geom is not None, f'No DeepSupportConvex found ' + \
            f'in {all_geoms}.'

        deep_support_network = deep_support_geom.network

        # Next load a random subset of the BundleSDF data.
        bsdf_dirs, bsdf_pts, bsdf_ds = file_utils.get_bundlesdf_geometry_data(
            self.config.data_config.asset_subdirectories,
            self.config.data_config.bundlesdf_id,
            iteration = int(self.config.data_config.tracker.split('_')[-1])
        )
        n_points = bsdf_pts.shape[0]
        n_random_points = min(1000, n_points)
        random_indices = torch.randperm(n_points)[:n_random_points]
        dirs = bsdf_dirs[random_indices]
        pts = bsdf_pts[random_indices]
        # scalars = bsdf_ds[random_indices]

        # # Compute the loss as absolute value difference on the scalar outputs.
        # loss = (scalars - deep_support_network.get_output(dirs)).abs()

        # Finally, compute the loss as L1 norm on the point locations.
        loss = torch.linalg.norm(pts - deep_support_network(dirs), dim=1)
        loss = loss.mean()
        return loss

    def write_to_wandb(self, epoch: int,
                       learned_system: MultibodyLearnableSystem,
                       statistics: Dict) -> None:
        """In addition to extracting and writing training progress summary via
        the parent :py:meth:`Experiment.write_to_wandb` method, also make a
        breakdown plot of loss contributions for the vision loss formulation.

        Args:
            epoch: Current epoch.
            learned_system: System being trained.
            statistics: Summary statistics for learning process.
        """
        assert self.wandb_manager is not None
        assert isinstance(learned_system, MultibodyLearnableSystem)

        # begin recording wall-clock logging time.
        start_log_time = time.time()

        # To save space on W&B storage, only generate comparison videos at first
        # and best epoch, the latter of which is implemented in
        # :meth:`_evaluation`.
        force_generate_videos = True if epoch == 0 else False

        epoch_vars, learned_system_summary = \
            self.build_epoch_vars_and_system_summary(
                statistics, learned_system,
                force_generate_videos=force_generate_videos
            )

        # Start computing individual loss components.
        # First get a batch sized portion of the shuffled training set.
        train_traj_set, _, _ = \
            self.learning_data_manager.get_updated_trajectory_sets()
        train_dataloader = DataLoader(
            train_traj_set.slices,
            batch_size=self.config.optimizer_config.batch_size.value,
            shuffle=True)

        # Calculate the average loss components.
        losses_pred, losses_comp, losses_pen, losses_diss = [], [], [], []
        losses_bsdf = []
        for xy_i in train_dataloader:
            # HACK:  Assumes 'state' key.
            x_i: Tensor = xy_i[0]['state']
            y_i: Tensor = xy_i[1]['state']

            x = x_i[..., -1, :]
            x_plus = y_i[..., 0, :]
            u = torch.zeros(x.shape[:-1] + (0,))

            loss_pred, loss_comp, loss_pen, loss_diss = \
                learned_system.calculate_contactnets_loss_terms(x, u, x_plus)
            loss_bsdf = self.bundlesdf_geometry_loss(learned_system)

            losses_pred.append(loss_pred.clone().detach())
            losses_comp.append(loss_comp.clone().detach())
            losses_pen.append(loss_pen.clone().detach())
            losses_diss.append(loss_diss.clone().detach())
            losses_bsdf.append(loss_bsdf.clone().detach())

        def really_weird_fix_for_cluster_only(list_of_tensors):
            """For some reason, on the cluster only, the last item in the loss
            lists can be a different shape than the rest of the items, and this
            results in an error with the ``sum(losses_pred)`` below.  For now,
            the fix (hack) is to just drop that last term.

            TODO:  Figure out what is going on.
            """
            if (len(list_of_tensors) > 1) and \
               (list_of_tensors[-1].shape != list_of_tensors[0].shape):
                    return list_of_tensors[:-1]
            return list_of_tensors

        losses_pred = really_weird_fix_for_cluster_only(losses_pred)
        losses_comp = really_weird_fix_for_cluster_only(losses_comp)
        losses_pen = really_weird_fix_for_cluster_only(losses_pen)
        losses_diss = really_weird_fix_for_cluster_only(losses_diss)
        losses_bsdf = really_weird_fix_for_cluster_only(losses_bsdf)

        # Calculate average and scale by hyperparameter weights.
        w_pred = learned_system.w_pred
        w_comp = learned_system.w_comp
        w_diss = learned_system.w_diss
        w_pen = learned_system.w_pen
        w_bsdf = learned_system.w_bsdf

        avg_loss_pred = w_pred*cast(Tensor, sum(losses_pred) \
                            / len(losses_pred)).mean()
        avg_loss_comp = w_comp*cast(Tensor, sum(losses_comp) \
                            / len(losses_comp)).mean()
        avg_loss_pen = w_pen*cast(Tensor, sum(losses_pen) \
                            / len(losses_pen)).mean()
        avg_loss_diss = w_diss*cast(Tensor, sum(losses_diss) \
                            / len(losses_diss)).mean()
        avg_loss_bsdf = w_bsdf*cast(Tensor, sum(losses_bsdf) \
                            / len(losses_bsdf)).mean()

        avg_loss_total = torch.sum(avg_loss_pred + avg_loss_comp + \
                                   avg_loss_pen + avg_loss_diss + avg_loss_bsdf)

        loss_breakdown = {'loss_total': avg_loss_total,
                          'loss_pred': avg_loss_pred,
                          'loss_comp': avg_loss_comp,
                          'loss_pen': avg_loss_pen,
                          'loss_diss': avg_loss_diss,
                          'loss_bsdf': avg_loss_bsdf}

        # Include the loss components into system summary.
        epoch_vars.update(loss_breakdown)

        # Overwrite the logging time.
        logging_duration = time.time() - start_log_time
        epoch_vars[LOGGING_DURATION] = logging_duration

        self.wandb_manager.update(epoch, epoch_vars,
                                  learned_system_summary.videos,
                                  learned_system_summary.meshes)

    def get_true_geometry_multibody_learnable_system(
            self) -> MultibodyLearnableSystem:
        """Overwritten for vision experiments to use any specified pre-computed
        functions for the continuous dynamics."""
        has_property = hasattr(self, 'true_geom_multibody_system')
        if not has_property or self.true_geom_multibody_system is None:
            oracle_system = self.get_oracle_system()
            dt = oracle_system.dt
            urdfs = oracle_system.urdfs

            self.true_geom_multibody_system = MultibodyLearnableSystem(
                init_urdfs=urdfs,
                dt=dt,
                loss_weights_dict={
                    'w_pred': self.config.learnable_config.w_pred,
                    'w_comp': self.config.learnable_config.w_comp,
                    'w_diss': self.config.learnable_config.w_diss,
                    'w_pen': self.config.learnable_config.w_pen,
                    'w_bsdf': self.config.learnable_config.w_bsdf},
                inertia_mode = self.config.learnable_config.inertia_mode,
                constant_bodies = self.config.learnable_config.constant_bodies,
                represent_geometry_as = \
                    self.config.learnable_config.represent_geometry_as,
                precomputed_functions = self.precomputed_functions)

        return self.true_geom_multibody_system

    def get_learned_system(self, _: Tensor) -> MultibodyLearnableSystem:
        """Overwritten for vision experiments to use any specified pre-computed
        functions for the continuous dynamics."""
        learnable_config = cast(MultibodyLearnableSystemConfig,
                                self.config.learnable_config)
        output_dir = file_utils.get_learned_urdf_dir(self.config.storage,
                                                     self.config.run_name)
        return MultibodyLearnableSystem(
            learnable_config.urdfs,
            self.config.data_config.dt,
            inertia_mode = learnable_config.inertia_mode,
            constant_bodies = learnable_config.constant_bodies,
            loss_weights_dict={
                'w_pred': learnable_config.w_pred,
                'w_comp': learnable_config.w_comp,
                'w_diss': learnable_config.w_diss,
                'w_pen': learnable_config.w_pen,
                'w_bsdf': learnable_config.w_bsdf},
            output_urdfs_dir=output_dir,
            represent_geometry_as=learnable_config.represent_geometry_as,
            pretrained_icnn_weights_filepath = \
                learnable_config.pretrained_icnn_weights_filepath,
                precomputed_functions = self.precomputed_functions)


class VisionRobotExperiment(VisionExperiment):
    """Class for vision experiments with robot interaction."""

    def __init__(self, config: VisionExperimentConfig) -> None:
        super().__init__(config)

    def get_loss_args(self,
                      x_past: TensorDictBase,
                      x_future: TensorDictBase,
                      system: System
                      ) -> Dict[str, Any]:
        """Extract the loss arguments from an input of past and future
        information, getting controls under the key 'robot_effort'."""
        x_u_xplus_dict = super().get_loss_args(x_past, x_future, system)

        # Get the control from the past state.
        n_horizon = x_past.shape[0]
        x_u_xplus_dict['u'] = x_past['robot_effort'].reshape(n_horizon, -1)

        return x_u_xplus_dict


"""Precomputed mass matrix and lagrangian forces expressions for robot
interaction vision experiments."""
from typing_extensions import Protocol
import types
import copy

class TensorCallable(Protocol):
    def __call__(self, *args: torch.Tensor) -> torch.Tensor: ...


def get_precomputed_mass_matrix_function(
        txt_function_directory: str) -> Callable[[Tensor], Tensor]:
    # Look for mass_matrix_{row}_{col}_func.txt.
    empty_row = [None] * 13
    matrix_of_functions = [copy.copy(empty_row) for _ in range(13)]
    for row in range(13):
        for col in range(13):
            func_txt_file = op.join(
                txt_function_directory, f'mass_matrix_{row}_{col}_func.txt')
            assert op.exists(func_txt_file), f'Need {func_txt_file} to exist.'

            with open(func_txt_file, 'r') as file:
                func_string = file.read()

            code = compile(func_string, f'tmp_M_{row}_{col}.py', 'single')
            func = cast(TensorCallable,
                        types.FunctionType(code.co_consts[0], globals()))

            matrix_of_functions[row][col] = func

    def mass_matrix_func(*torch_args):  # q, inertia
        batch_dims = torch_args[0].shape[:-1]
        mass_matrix = torch.zeros(batch_dims + (13, 13))
        for row in range(13):
            for col in range(13):
                mass_matrix[..., row, col] = \
                    matrix_of_functions[row][col](*torch_args)
        return mass_matrix

    return mass_matrix_func

def get_precomputed_lagrangian_forces_function(
        txt_function_directory: str) -> Callable[[Tensor], Tensor]:
    # Look for lagrangian_forces_{row}_func.txt.
    vector_of_functions = [None] * 13
    for row in range(13):
        func_txt_file = op.join(
            txt_function_directory, f'lagrangian_forces_{row}_func.txt')
        assert op.exists(func_txt_file), f'Need {func_txt_file} to exist.'

        with open(func_txt_file, 'r') as file:
            func_string = file.read()

        code = compile(func_string, f'tmp_lagr_{row}.py', 'single')
        func = cast(TensorCallable,
                    types.FunctionType(code.co_consts[0], globals()))

        vector_of_functions[row] = copy.deepcopy(func)

    def lagrangian_forces_func(*torch_args):  # q, v, u, inertia
        batch_dims = torch_args[0].shape[:-1]
        lagrangian_forces = torch.zeros(batch_dims + (13,))
        for row in range(13):
            lagrangian_forces[..., row] = vector_of_functions[row](*torch_args)
        return lagrangian_forces

    return lagrangian_forces_func
