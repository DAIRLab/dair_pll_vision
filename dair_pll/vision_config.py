"""File with class extensions and variable definitions for BundleSDF/PLL vision
project."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, cast, Dict, Union, Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pdb
import re
import time
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tensordict.tensordict import TensorDictBase

from dair_pll import file_utils, vis_utils
from dair_pll.data_config import DataConfig
from dair_pll.dataset_management import ExperimentDataManager, TrajectorySet
from dair_pll.drake_experiment import DrakeMultibodyLearnableExperimentConfig, \
    DrakeMultibodyLearnableExperiment, MultibodyLosses, \
    MultibodyLearnableSystemConfig
from dair_pll.experiment import LossCallbackCallable, TrainingState, \
    LOGGING_DURATION, StatisticsDict, StatisticsValue, TRAIN_SET, LOSS_NAME, \
    AVERAGE_TAG, TRAIN_TIME_SETS, EVALUATION_VARIABLES, LEARNED_SYSTEM_NAME, \
    ALL_DURATIONS, TARGET_NAME, MAX_SAVED_TRAJECTORIES, ALL_SETS, \
    ORACLE_SYSTEM_NAME
from dair_pll.file_utils import EXPORT_POINTS_DEFAULT_NAME, \
    EXPORT_DIRECTIONS_DEFAULT_NAME, \
    EXPORT_FORCES_DEFAULT_NAME, \
    EXPORT_STATES_DEFAULT_NAME, \
    EXPORT_TOSS_FRAME_IDX_DEFAULT_NAME, \
    EXPORT_ALL_FORCES_DEFAULT_NAME, \
    EXPORT_PHIS_DEFAULT_NAME, \
    EXPORT_JACOBIANS_DEFAULT_NAME
from dair_pll.geometry import DeepSupportConvex
from dair_pll.system import System, SystemSummary
from dair_pll.multibody_learnable_system import MultibodyLearnableSystem, \
    ContactNetsLossReturnType
from dair_pll.multibody_terms import PRECOMPUTED_FUNCTION_KEY, \
    PRECOMPUTED_FUNCTION_STATES_KEY
from dair_pll.tensor_utils import pbmm


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
                  'vision_mug', 'vision_oatly', 'vision_pinkcan',
                  'vision_stapler', 'vision_styrofoam', 'vision_toothpaste',
                  'vision_robot_bakingbox_sticky_A', 'vision_robot_bakingbox',
                  'vision_robot_greencan', 'vision_robot_oatly',
                  'vision_robot_stapler', 'vision_robot_milk'
                  ]


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

        # Get precomputed functions each in the format:
        # {'function': callable, 'state_names': list[str]}
        precomputed_functions = {}
        dirs = config.learnable_config.precomputed_function_directories
        if 'mass_matrix' in dirs.keys():
            precomputed_functions['mass_matrix'] = \
                get_precomputed_mass_matrix_function_and_states(
                    dirs['mass_matrix'])
        if 'lagrangian_forces' in dirs.keys():
            precomputed_functions['lagrangian_forces'] = \
                get_precomputed_lagrangian_forces_function_and_states(
                    dirs['lagrangian_forces'])
        self.precomputed_functions = precomputed_functions

        # Store the optional location at which Drake Pytorch expressions that
        # are computed for these systems will be exported.
        self.export_drake_pytorch_dir = \
            config.learnable_config.export_drake_pytorch_dir

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
        # bundlesdf_loss = torch.zeros_like(contactnets_loss)
        # print("Warning: bundlesdf_loss is currently disabled.")

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
        bsdf_dirs, bsdf_pts, _bsdf_ds = file_utils.get_bundlesdf_geometry_data(
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
            x_i: Tensor = xy_i[0]
            y_i: Tensor = xy_i[1]

            loss_pred, loss_comp, loss_pen, loss_diss = \
                learned_system.calculate_contactnets_loss_terms(
                    **self.get_loss_args(x_i, y_i, learned_system))
            loss_bsdf = self.bundlesdf_geometry_loss(learned_system)
            # loss_bsdf = torch.zeros_like(loss_pred)
            # print("Warning: bundlesdf_loss is currently disabled in write_to_wandb.")

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

            # Ensure any mesh geometries preserve the vertex set.
            represent_geometry_as = 'polygon' if \
                self.config.learnable_config.represent_geometry_as == 'mesh' \
                else self.config.learnable_config.represent_geometry_as

            self.true_geom_multibody_system = MultibodyLearnableSystem(
                init_urdfs=urdfs,
                dt=dt,
                loss_weights_dict={
                    'w_pred': self.config.learnable_config.w_pred,
                    'w_comp': self.config.learnable_config.w_comp,
                    'w_diss': self.config.learnable_config.w_diss,
                    'w_pen': self.config.learnable_config.w_pen,
                    'w_bsdf': self.config.learnable_config.w_bsdf},
                learnable_body_dict = \
                    self.config.learnable_config.learnable_body_dict,
                represent_geometry_as = represent_geometry_as,
                precomputed_functions=self.precomputed_functions,
                export_drake_pytorch_dir=self.export_drake_pytorch_dir)

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
            learnable_body_dict=learnable_config.learnable_body_dict,
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
            precomputed_functions=self.precomputed_functions,
            export_drake_pytorch_dir = self.export_drake_pytorch_dir)


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

    def evaluate_systems_on_sets(
            self, systems: Dict[str, System],
            sets: Dict[str, TrajectorySet]) -> StatisticsDict:
        """Overwritten to exclude some metrics that aren't well-defined (yet)
        for robot interaction epxeriments."""
        stats = {}  # type: StatisticsDict

        def to_json(possible_tensor: Union[float, List, Tensor]) -> \
                StatisticsValue:
            """Converts tensor to :class:`~np.ndarray`, which enables saving
            stats as json."""
            if isinstance(possible_tensor, list):
                return [to_json(value) for value in possible_tensor]
            if torch.is_tensor(possible_tensor):
                tensor = cast(Tensor, possible_tensor)
                return tensor.detach().cpu().numpy()

            assert isinstance(possible_tensor, float)
            return possible_tensor

        for set_name, trajectory_set in sets.items():
            # Avoid error case if one of the sets is empty (e.g. test set).
            if trajectory_set.indices.shape[0] == 0:
                continue

            trajectories = trajectory_set.trajectories
            n_saved_trajectories = min(MAX_SAVED_TRAJECTORIES,
                                       len(trajectories))
            slices_loader = DataLoader(trajectory_set.slices,
                                       batch_size=128,
                                       shuffle=False)

            # Skip velocity square error.

            for system_name, system in systems.items():
                # Skip prediction loss.

                # The training loss on the training set will get added to the
                # stats dictionary in per_epoch_evaluation.  The below computes
                # the same loss metric but for the other sets (val/test).
                if set_name != TRAIN_SET:
                    model_loss_list = []
                    for batch_x, batch_y in slices_loader:
                        model_loss_list.append(
                            self.loss_callback(batch_x, batch_y, system, True))
                    model_loss = torch.cat(model_loss_list)
                    loss_name = f'{set_name}_{system_name}_{LOSS_NAME}'
                    stats[loss_name] = to_json(model_loss)

                # Skip trajectory predictions, but store the input trajectories.
                if system_name == LEARNED_SYSTEM_NAME:
                    trajectories = [t.unsqueeze(0) for t in trajectories]

                    # Get the state in tensor form.
                    x = [system.construct_state_tensor(xi) for xi in \
                         trajectories]

                    t_skip = self.config.data_config.slice_config.t_skip
                    t_begin = t_skip + 1
                    targets = [x_i[..., t_begin:, :].squeeze(0) for x_i in x]
                    stats[f'{set_name}_{system_name}_{TARGET_NAME}'] = \
                        to_json(targets[:n_saved_trajectories])

                # Skip extra metrics and auxillary losses.

        summary_stats = {}  # type: StatisticsDict
        for key, stat in stats.items():
            if isinstance(stat, np.ndarray):
                if len(stat) > 0:
                    if isinstance(stat[0], float):
                        summary_stats[f'{key}_{AVERAGE_TAG}'] = np.average(stat)

        stats.update(summary_stats)
        return stats

    def construct_trajectory_for_comparison_vis(
            self, target_trajectory: Tensor, prediction_trajectory: Tensor
    ) -> Tensor:
        assert target_trajectory.shape == prediction_trajectory.shape

        space = self.get_drake_system().space

        # HACK:  hard-code the state ordering.
        target_q, target_v = space.q_v(target_trajectory)
        _world_q, target_robot_q, target_object_q = space.q_split(target_q)
        _world_v, target_robot_v, target_object_v = space.v_split(target_v)

        pred_q, pred_v = space.q_v(prediction_trajectory)
        _world_q, pred_robot_q, pred_object_q = space.q_split(pred_q)
        _world_v, pred_robot_v, pred_object_v = space.v_split(pred_v)

        assert target_robot_q.shape[-1] == target_object_q.shape[-1] == 7
        assert target_robot_v.shape[-1] == 7
        assert target_object_v.shape[-1] == 6

        # The visualization system should have both robots first and
        # both objects second.
        return torch.cat(
            (target_robot_q, pred_robot_q, target_object_q, pred_object_q,
             target_robot_v, pred_robot_v, target_object_v, pred_object_v), -1)

    def base_and_learned_comparison_summary(
            self, statistics: Dict, learned_system: System,
            force_generate_videos: bool = False) -> SystemSummary:
        r"""Extracts a :py:class:`~dair_pll.system.SystemSummary` that compares
        the base system to the learned system.

        For Drake-based experiments, this comparison is implemented as
        overlaid videos of corresponding ground-truth and predicted
        trajectories. The nature of this video is described further in
        :py:mod:`dair_pll.vis_utils`\ .

        Additionally, manually defined trajectories are used to show the learned
        geometries.  This is particularly useful for more expressive geometry
        types like meshes.

        Args:
            statistics: Dictionary of training statistics.
            learned_system: Most updated version of learned system during
              training.
            force_generate_videos: Whether to force generate videos for
              comparison, even if the experiment's config says to skip.  This is
              useful for generating videos at the first and last epochs.

        Returns:
            Summary containing overlaid video(s).
        """
        # return SystemSummary(scalars={}, videos={}, meshes={})
        if (not force_generate_videos) and \
            (not self.config.generate_video_predictions_throughout) and \
            (not self.config.generate_video_geometries_throughout):
            return SystemSummary(scalars={}, videos={}, meshes={})

        # Include all of the base and learned comparison videos from the parent
        # class.
        summary = super().base_and_learned_comparison_summary(
            statistics, learned_system, force_generate_videos)

        # Add an input visualization to see the trajectory with the robot.
        visualization_system = self.get_visualization_system(learned_system)
        videos = {}

        for traj_num in [0]:
            for set_name in ['train', 'valid']:
                target_key = f'{set_name}_{LEARNED_SYSTEM_NAME}_{TARGET_NAME}'
                if not target_key in statistics:
                    continue
                target_trajectory = Tensor(statistics[target_key][traj_num])

                # The visualization system should have the same trajectory for
                # both the learned and predicted systems.
                visualization_trajectory = \
                    self.construct_trajectory_for_comparison_vis(
                        target_trajectory, target_trajectory)

                video, framerate = vis_utils.visualize_trajectory(
                    visualization_system, visualization_trajectory)
                videos[f'{set_name}_input_trajectory_{traj_num}'] = \
                    (video, framerate)

        # Add the input trajectory visualization to the summary.
        summary.videos.update(videos)

        return summary

    def build_epoch_vars_and_system_summary(self, statistics: Dict,
            learned_system: System, force_generate_videos: bool = False
        ) -> Tuple[Dict, SystemSummary]:
        """Build epoch variables and system summary for learning process.

        Args:
            statistics: Summary statistics for learning process.
            learned_system: System being trained.
            force_generate_videos: Whether to force generate videos for
              comparison, even if the experiment's config says to skip.  This is
              useful for generating videos at the first and last epochs.

        Returns:
            Dictionary of scalars to log.
            System summary.
        """
        # begin recording wall-clock logging time.
        start_log_time = time.time()

        epoch_vars = {}
        for stats_set in TRAIN_TIME_SETS:
            for variable in EVALUATION_VARIABLES:
                var_key = f'{stats_set}_{LEARNED_SYSTEM_NAME}' + \
                          f'_{variable}_{AVERAGE_TAG}'
                if var_key in statistics:
                    epoch_vars[f'{stats_set}_{variable}'] = statistics[var_key]

        learned_system_summary = learned_system.summary(statistics)

        # Include comparison summary, which should only create the geometry
        # inspection video since no predictions to visualize.
        comparison_summary = self.base_and_learned_comparison_summary(
            statistics, learned_system,
            force_generate_videos=force_generate_videos)

        epoch_vars.update(learned_system_summary.scalars)
        logging_duration = time.time() - start_log_time
        statistics[LOGGING_DURATION] = logging_duration
        epoch_vars.update(
            {duration: statistics[duration] for duration in ALL_DURATIONS})

        epoch_vars.update(comparison_summary.scalars)
        learned_system_summary.videos.update(comparison_summary.videos)
        learned_system_summary.meshes.update(comparison_summary.meshes)

        return epoch_vars, learned_system_summary

    def debug_training(self, learned_system: System,
                       store_to_file: bool = False) -> None:
        """Debugging function for vision robot experiments."""
        # First load all the BundleSDF exported data.
        run_name = self.config.run_name
        storage_name = self.config.storage
        output_dir = file_utils.geom_for_bsdf_dir(storage_name, run_name)

        normal_forces_stacked = torch.load(
            op.join(output_dir, EXPORT_FORCES_DEFAULT_NAME),
            weights_only=True).detach()
        support_points_stacked = torch.load(
            op.join(output_dir, EXPORT_POINTS_DEFAULT_NAME),
            weights_only=True).detach()
        support_directions_stacked = torch.load(
            op.join(output_dir, EXPORT_DIRECTIONS_DEFAULT_NAME),
            weights_only=True).detach()
        states_stacked = torch.load(
            op.join(output_dir, EXPORT_STATES_DEFAULT_NAME),
            weights_only=True).detach()
        all_forces_stacked = torch.load(
            op.join(output_dir, EXPORT_ALL_FORCES_DEFAULT_NAME),
            weights_only=True).detach()
        phis_stacked = torch.load(
            op.join(output_dir, EXPORT_PHIS_DEFAULT_NAME),
            weights_only=True).detach()
        jacobians_stacked = torch.load(
            op.join(output_dir, EXPORT_JACOBIANS_DEFAULT_NAME),
            weights_only=True).detach()

        # Split by contact:  [timestep_i, contact_i, ...]
        n_contacts = 6
        n_timesteps = normal_forces_stacked.shape[0] // n_contacts
        n_state = learned_system.space.n_x
        n_velocity = learned_system.space.n_v
        assert normal_forces_stacked.ndim == 1 and \
            normal_forces_stacked.shape[0] / n_contacts == \
            normal_forces_stacked.shape[0] // n_contacts

        normal_forces = torch.zeros((0, n_contacts))
        support_points = torch.zeros((0, n_contacts, 3))
        support_directions = torch.zeros((0, n_contacts, 3))
        states = torch.zeros((0, n_contacts, n_state))
        all_forces = torch.zeros((0, n_contacts, 3))
        phis = torch.zeros((0, n_contacts))
        jacobians = torch.zeros((0, n_contacts, 3, n_velocity))

        forces_on_object = torch.zeros((0, n_contacts, 3))
        # unscaled_losses = torch.zeros((0, 4))

        for i in range(n_timesteps):
            idx1 = i*n_contacts
            idx2 = (i+1)*n_contacts
            normal_forces = torch.cat(
                (normal_forces,
                 normal_forces_stacked[idx1:idx2].unsqueeze(0)
                ), dim=0)
            support_points = torch.cat(
                (support_points,
                 support_points_stacked[idx1:idx2].unsqueeze(0)
                ), dim=0)
            support_directions = torch.cat(
                (support_directions,
                 support_directions_stacked[idx1:idx2].unsqueeze(0)
                ), dim=0)
            states = torch.cat(
                (states,
                 states_stacked[idx1:idx2].unsqueeze(0)
                ), dim=0)
            all_forces = torch.cat(
                (all_forces,
                 all_forces_stacked[idx1:idx2].unsqueeze(0)
                ), dim=0)
            phis = torch.cat(
                (phis,
                 phis_stacked[idx1:idx2].unsqueeze(0)
                ), dim=0)
            jacobians = torch.cat(
                (jacobians,
                 jacobians_stacked[idx1:idx2].unsqueeze(0)
                ), dim=0)

            # Compute the forces on the object.
            J = jacobians_stacked[idx1:idx2]    # (n_lambda, 3, n_velocity)
            contact_force = all_forces_stacked[idx1:idx2].unsqueeze(-1)
                                                # (n_lambda, 3, 1)
            generalized_forces = pbmm(
                J.transpose(-1, -2), contact_force).reshape(
                    n_contacts, n_velocity)       # (n_lambda, n_velocity)
            force_on_object = generalized_forces[:, -3:]
            forces_on_object = torch.cat(
                (forces_on_object,
                 force_on_object.unsqueeze(0)
            ), dim=0)

        # Rebuild the impulses vector of shape (N, n_contact*3).
        normal_z_forces = all_forces[1:, :, 0]
        tangential_x_forces = all_forces[1:, :, 1]
        tangential_y_forces = all_forces[1:, :, 2]

        tx_idx = [idx for idx in range(n_contacts, 3*n_contacts, 2)]
        ty_idx = [idx for idx in range(n_contacts+1, 3*n_contacts, 2)]

        impulses = torch.zeros((n_timesteps-1, 3*n_contacts))
        for i in range(n_timesteps-1):
            impulses[i, :n_contacts] = normal_z_forces[i, :] * 1.0/30
            impulses[i, tx_idx] = tangential_x_forces[i, :] * 1.0/30
            impulses[i, ty_idx] = tangential_y_forces[i, :] * 1.0/30
        impulses = impulses.unsqueeze(-1)

        # Compute the unscaled loss terms.
        states_prev = states[:-1, 0, :].reshape(n_timesteps-1, n_state)
        states_next = states[1:, 0, :].reshape(n_timesteps-1, n_state)
        control = torch.zeros((n_timesteps-1, 7))
        Q, q_pred, q_comp, q_diss, c_pen, c_pred = \
            learned_system.calculate_contactnets_loss_terms(
                states_prev, control, states_next,
                return_type=ContactNetsLossReturnType.UNSCALED_LOSS_STRUCTURE)
        l_pred = 0.5 * pbmm(impulses.transpose(-1, -2), pbmm(Q, impulses)) \
            + pbmm(impulses.transpose(-1, -2), q_pred) + c_pred
        l_pred = l_pred.squeeze()
        l_comp = pbmm(impulses.transpose(-1, -2), q_comp).squeeze()
        l_diss = pbmm(impulses.transpose(-1, -2), q_diss).squeeze()
        l_pen = c_pen.squeeze()

        times = torch.arange(normal_forces.shape[0]) * (1.0/30)

        # Visualize the forces and signed distances.
        if not store_to_file:
            plt.ion()
        fig, ax = plt.subplots(4, 2, figsize=(15, 15), sharex='all')
        for i in range(n_contacts):
            ax[0, 0].plot(times, normal_forces[:, i], linewidth=6-i,
                          label=f'Contact {i}')
            ax[1, 0].plot(times, phis[:, i], linewidth=6-i,
                          label=f'Contact {i}')
        ax[2, 0].plot(times, states[:, 0, -3], label='Object v_x')
        ax[2, 0].plot(times, states[:, 0, -2], label='Object v_y')
        ax[2, 0].plot(times, states[:, 0, -1], label='Object v_z')

        for i in range(n_contacts):
            ax[0, 1].plot(times, forces_on_object[:, i, 0], linewidth=6-i,
                          label=f'Contact {i}')
            ax[1, 1].plot(times, forces_on_object[:, i, 1], linewidth=6-i,
                          label=f'Contact {i}')
            ax[2, 1].plot(times, forces_on_object[:, i, 2], linewidth=6-i,
                          label=f'Contact {i}')
        total_forces_xyz = torch.sum(forces_on_object, dim=1)
        ax[0, 1].plot(times, total_forces_xyz[:, 0], linestyle='--', color='k',
                      label='Total')
        ax[1, 1].plot(times, total_forces_xyz[:, 1], linestyle='--', color='k',
                      label='Total')
        ax[2, 1].plot(times, total_forces_xyz[:, 2], linestyle='--', color='k',
                      label='Total')

        ax[3, 0].plot(times[1:], l_pred.detach().numpy(), linewidth=4,
                      label='Prediction')
        ax[3, 0].plot(times[1:], l_comp.detach().numpy(), linewidth=3,
                      label='Complementarity')
        ax[3, 0].plot(times[1:], l_diss.detach().numpy(), linewidth=2,
                      label='Dissipation')
        ax[3, 0].plot(times[1:], l_pen.detach().numpy(), linewidth=1,
                      label='Penetration')

        ax[3, 0].set_xlabel('Time [s]')
        ax[3, 1].set_xlabel('Time [s]')
        for ax_i in ax.flatten():
            ax_i.tick_params(labelbottom=True)
        ax[0, 0].set_ylabel('Normal Forces [N]')
        ax[1, 0].set_ylabel('Signed Distances [m]')
        ax[2, 0].set_ylabel('Object Velocities [m/s]')
        ax[3, 0].set_ylabel('Unscaled Loss Terms')
        ax[0, 1].set_ylabel('Forces on Object X [N]')
        ax[1, 1].set_ylabel('Forces on Object Y [N]')
        ax[2, 1].set_ylabel('Forces on Object Z [N]')
        # ax[0, 0].set_title('Normal Forces')
        # ax[1, 0].set_title('Signed Distances')
        # ax[2, 0].set_title('Object Velocities')
        ax[1, 0].legend()
        ax[2, 0].legend()
        ax[3, 0].legend()
        ax[1, 1].legend()
        fig.suptitle(f'Debugging {run_name}')

        if store_to_file:
            # Save the figure to file.
            filepath = file_utils.debug_plot_filepath(storage_name, run_name)
            fig.savefig(filepath)
            print(f'Saved debug plot to {filepath}')
        else:
            pdb.set_trace()




"""Precomputed mass matrix and lagrangian forces expressions for robot
interaction vision experiments."""
from typing_extensions import Protocol
import types
import copy

class TensorCallable(Protocol):
    def __call__(self, *args: torch.Tensor) -> torch.Tensor: ...


def get_precomputed_mass_matrix_function_and_states(txt_function_directory: str
) -> Dict[str, Union[ List[str], Callable[[Tensor], Tensor] ]]:
    # Get the expected system state names at mass_matrix_state_names.txt.
    state_names_file = op.join(
        txt_function_directory, 'mass_matrix_state_names.txt')
    assert op.exists(state_names_file), f'Need {state_names_file} to exist.'
    with open(state_names_file, 'r') as file:
        state_names = file.read().splitlines()

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

    return {PRECOMPUTED_FUNCTION_KEY: mass_matrix_func,
            PRECOMPUTED_FUNCTION_STATES_KEY: state_names}


def get_precomputed_lagrangian_forces_function_and_states(
        txt_function_directory: str
) -> Dict[str, Union[ List[str], Callable[[Tensor], Tensor] ]]:

    # Get the expected system state names at lagrangian_forces_state_names.txt.
    state_names_file = op.join(
        txt_function_directory, 'lagrangian_forces_state_names.txt')
    assert op.exists(state_names_file), f'Need {state_names_file} to exist.'
    with open(state_names_file, 'r') as file:
        state_names = file.read().splitlines()

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

    return {PRECOMPUTED_FUNCTION_KEY: lagrangian_forces_func,
            PRECOMPUTED_FUNCTION_STATES_KEY: state_names}
