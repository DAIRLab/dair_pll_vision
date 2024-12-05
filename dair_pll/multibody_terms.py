"""Mathematical implementation of multibody dynamics terms calculations.

This file implements the :py:class:`MultibodyTerms` type, which interprets a
list of urdfs as a learnable Lagrangian system with contact, taking the state
space from the corresponding :py:class:`MultibodyPlantDiagram` as a given, and
interpreting the various inertial and geometric terms stored within it as
initial conditions of learnable parameters.

Multibody dynamics can be derived from four functions of state [q,v]:

    * M(q), the generalized mass-matrix
    * F(q), the non-contact/Lagrangian force terms.
    * phi(q), the signed distance between collision candidates.
    * J(q), the contact-frame velocity Jacobian between collision candidates.

The first two terms depend solely on state and inertial properties,
and parameterize the contact-free Lagrangian dynamics as::

    dv/dt = (M(q) ** (-1)) * F(q)

These terms are accordingly encapsulated in a :py:class:`LagrangianTerms`
instance.

The latter two terms depend solely on the geometry of bodies coming into
contact, and are encapsulated in a :py:class:`ContactTerms` instance.

For both sets of terms, we derive their functional form either directly or in
part through symbolic analysis of the :py:class:`MultibodyPlant` of the
associated :py:class:`MultibodyPlantDiagram`. The :py:class:`MultibodyTerms`
object manages the symbolic calculation and has corresponding
:py:class:`LagrangianTerms` and :py:class:`ContactTerms` members.
"""
from typing import List, Tuple, Callable, Dict, cast, Optional, Union
from dataclasses import dataclass

import drake_pytorch  # type: ignore
import numpy as np
import os.path as op
import torch
import pdb

from pydrake.geometry import SceneGraphInspector, GeometryId  # type: ignore
from pydrake.multibody.plant import MultibodyPlant_  # type: ignore
from pydrake.multibody.tree import JacobianWrtVariable  # type: ignore
from pydrake.multibody.tree import ModelInstanceIndex  # type: ignore
from pydrake.multibody.tree import SpatialInertia_, UnitInertia_, \
                                   RotationalInertia_  # type: ignore
from pydrake.symbolic import Expression, Variable  # type: ignore
from pydrake.symbolic import MakeVectorVariable, Jacobian  # type: ignore
from pydrake.systems.framework import Context  # type: ignore
from scipy.spatial.transform import Rotation
from torch import Tensor
from torch.nn import Module, ModuleList, Parameter, ParameterList

from dair_pll import drake_utils, file_utils
from dair_pll.drake_utils import DrakeBody
from dair_pll.deep_support_function import extract_mesh_from_support_function, \
    get_mesh_summary_from_polygon
from dair_pll.drake_state_converter import DrakeStateConverter
from dair_pll.drake_utils import MultibodyPlantDiagram
from dair_pll.geometry import GeometryCollider, \
    PydrakeToCollisionGeometryFactory, \
    CollisionGeometry, DeepSupportConvex, Polygon, Box, \
    Plane, _NOMINAL_HALF_LENGTH
from dair_pll.inertia import InertialParameterConverter
from dair_pll.system import MeshSummary
from dair_pll.tensor_utils import (pbmm, deal, spatial_to_point_jacobian)
import matplotlib.pyplot as plt
from tempfile import TemporaryDirectory
import os
from tqdm import tqdm

ConfigurationInertialCallback = Callable[[Tensor, Tensor], Tensor]
StateInputInertialCallback = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]

CENTER_OF_MASS_DOF = 3
INERTIA_TENSOR_DOF = 6
DEFAULT_SIMPLIFIER = drake_pytorch.Simplifier.QUICKTRIG

PRECOMPUTED_FUNCTION_KEY = 'function'
PRECOMPUTED_FUNCTION_STATES_KEY = 'state_names'


@dataclass
class LearnableBodySettings:
    """Class to specify which body parameters to learn"""
    inertia_mass: bool = False
    inertia_com: bool = False
    inertia_moments_products: bool = False
    geometry: bool = False
    friction: bool = False


def are_body_properties_learnable(
        body_name: str, learnable_body_dict: Dict[str, LearnableBodySettings]
) -> Tuple[bool, bool, bool]:
    """Determines if a body's inertial, geometric, and frictional properties are
    learnable, independently, from a dictionary of learnable body settings."""
    if body_name in learnable_body_dict.keys():
        body_settings = learnable_body_dict[body_name]
        learn_inertia = body_settings.inertia_mass or \
            body_settings.inertia_com or \
            body_settings.inertia_moments_products
        learn_geometry = body_settings.geometry
        learn_friction = body_settings.friction
        return learn_inertia, learn_geometry, learn_friction
    else:
        return False, False, False


# noinspection PyUnresolvedReferences
def init_symbolic_plant_context_and_state(
    plant_diagram: MultibodyPlantDiagram
) -> Tuple[MultibodyPlant_[Expression], Context, np.ndarray, np.ndarray]:
    """Generates a symbolic interface for a :py:class:`MultibodyPlantDiagram`.

    Generates a new Drake ``Expression`` data type state in
    :py:class:`StateSpace` format, and sets this state inside a new context for
    a symbolic version of the diagram's :py:class:`MultibodyPlant`.

    Args:
        plant_diagram: Drake MultibodyPlant diagram to convert to symbolic.

    Returns:
        New symbolic plant.
        New plant's context, with symbolic states set.
        (n_q,) symbolic :py:class:`StateSpace` configuration.
        (n_v,) symbolic :py:class:`StateSpace` velocity.
    """
    plant = plant_diagram.plant.ToSymbolic()
    space = plant_diagram.space
    context = plant.CreateDefaultContext()

    # :py:class:`StateSpace` representation of Plant's state.
    q = MakeVectorVariable(plant.num_positions(), 'q', Variable.Type.CONTINUOUS)
    v = MakeVectorVariable(plant.num_velocities(), 'v',
                           Variable.Type.CONTINUOUS)
    x = np.concatenate([q, v], axis=-1)

    # Set :py:class:`StateSpace` symbolic state inside
    DrakeStateConverter.state_to_context(plant, context, x,
                                         plant_diagram.model_ids, space)
    return plant, context, q, v


class LagrangianTerms(Module):
    """Container class for non-contact/Lagrangian dynamics terms.

    Accepts batched pytorch callback functions for M(q) and F(q) and related
    contact terms in ``theta`` format (see ``inertia.py``).
    """
    mass_matrix: Optional[ConfigurationInertialCallback]
    lagrangian_forces: Optional[StateInputInertialCallback]
    body_parameters: ParameterList
    inertial_parameters: Tensor

    def __init__(self, plant_diagram: MultibodyPlantDiagram,
                 learnable_body_dict: Dict[str, LearnableBodySettings] = {},
                 precomputed_functions: Dict[str, Union[List[str],Callable]]={},
                 export_drake_pytorch_dir: str = None
                 ) -> None:
        """Inits :py:class:`LagrangianTerms` with prescribed parameters and
        functional forms.

        Args:
            plant_diagram: Drake MultibodyPlant diagram to extract terms from.
            learnable_body_dict: dictionary of which body parameters to learn.
            precomputed_functions: Dictionary of precomputed functions.  Keys
                that will be considered are 'mass_matrix' and
                'lagrangian_forces'.  The values at those keys are nested
                dictionaries with keys 'function' with the callable and
                'state_names' with a list of strings for the plant's state names
                that were used when the function was pre-computed.  The state
                names are checked to match the state names of the newly created
                plant.
            export_drake_pytorch_dir: The folder in which exported elements of
                the mass matrix and lagrangian force expressions will be saved.
                If provided, the code terminates after the export.
        """
        super().__init__()

        if export_drake_pytorch_dir is not None:
            precomputed_functions = {}

        plant, context, q, v = init_symbolic_plant_context_and_state(
            plant_diagram)
        gamma = Jacobian(plant.GetVelocities(context), v)

        body_param_tensors, learnable_body_variables, bodies, \
            learnable_body_idx = \
                LagrangianTerms.extract_body_parameters_and_variables(
                    plant, plant_diagram.model_ids, context,
                    learnable_body_dict=learnable_body_dict)

        if 'mass_matrix' not in precomputed_functions.keys():
            mass_matrix_expression = gamma.T @ \
                plant.CalcMassMatrixViaInverseDynamics(context) @ gamma

            print(f'\nMAKING MASS DRAKE PYTORCH EXPRESSION\n')
            if export_drake_pytorch_dir is not None:
                file_utils.assure_created(export_drake_pytorch_dir)
                for row in range(13):
                    for col in range(13):
                        print(f'Printing {row=}, {col=}')

                        # Save the Drake expression.
                        with open(
                            op.join(export_drake_pytorch_dir,
                                f'mass_matrix_{row}_{col}.txt'), 'w') as f:
                            f.write(str(mass_matrix_expression[row, col]))

                        # Save the pytorch function string.
                        _, func_string = drake_pytorch.sym_to_pytorch(
                            mass_matrix_expression[row, col],
                            q,
                            learnable_body_variables,
                            simplify_computation=DEFAULT_SIMPLIFIER)
                        with open(
                            op.join(export_drake_pytorch_dir,
                                f'mass_matrix_{row}_{col}_func.txt'), 'w') as f:
                            f.write(func_string)

            else:
                self.mass_matrix, _ = drake_pytorch.sym_to_pytorch(
                    mass_matrix_expression,
                    q,
                    learnable_body_variables,
                    simplify_computation=DEFAULT_SIMPLIFIER)

        else:
            print(f'Using pre-computed mass_matrix expression.')
            expected_state_names = precomputed_functions['mass_matrix'][
                PRECOMPUTED_FUNCTION_STATES_KEY]
            assert expected_state_names == plant.GetStateNames(), \
                f'Precomputed mass matrix uses {expected_state_names=} but ' + \
                f'plant has {plant.GetStateNames()}.'

            self.mass_matrix = precomputed_functions['mass_matrix'][
                PRECOMPUTED_FUNCTION_KEY]

        if 'lagrangian_forces' not in precomputed_functions.keys():
            u = MakeVectorVariable(plant.num_actuated_dofs(), 'u',
                                   Variable.Type.CONTINUOUS)
            drake_forces_expression = -plant.CalcBiasTerm(
                context) + plant.MakeActuationMatrix(
                ) @ u + plant.CalcGravityGeneralizedForces(context)

            lagrangian_forces_expression = gamma.T @ drake_forces_expression

            print(f'\nMAKING CONTINUOUS DYNAMICS DRAKE PYTORCH EXPRESSION\n')
            if export_drake_pytorch_dir is not None:
                for row in range(13):
                    print(f'Printing {row=}')

                    # Save the Drake expression.
                    with open(
                        op.join(export_drake_pytorch_dir,
                            f'lagrangian_forces_{row}.txt'), 'w') as f:
                        f.write(str(lagrangian_forces_expression[row]))

                    # Save the pytorch function string.
                    _, func_string = drake_pytorch.sym_to_pytorch(
                        lagrangian_forces_expression[row],
                        q, v, u,
                        learnable_body_variables,
                        simplify_computation=DEFAULT_SIMPLIFIER)
                    with open(
                        op.join(export_drake_pytorch_dir,
                            f'lagrangian_forces_{row}_func.txt'), 'w') as f:
                        f.write(func_string)
                exit()

            else:
                self.lagrangian_forces, _ = drake_pytorch.sym_to_pytorch(
                    lagrangian_forces_expression,
                    q,
                    v,
                    u,
                    learnable_body_variables,
                    simplify_computation=DEFAULT_SIMPLIFIER)

        else:
            print(f'Using pre-computed lagrangian_forces expression.')
            expected_state_names = precomputed_functions['lagrangian_forces'][
                PRECOMPUTED_FUNCTION_STATES_KEY]
            assert expected_state_names == plant.GetStateNames(), \
                f'Precomputed lagrangian forces use {expected_state_names=}' + \
                f' but plant has {plant.GetStateNames()}.'

            self.lagrangian_forces = precomputed_functions['lagrangian_forces'][
                PRECOMPUTED_FUNCTION_KEY]

        # pylint: disable=E1103
        self.body_parameters = ParameterList()
        
        for body_param_tensor, body in zip(body_param_tensors, bodies):
            inertia_learnable, _, _ = are_body_properties_learnable(
                body.name(), learnable_body_dict)
            if inertia_learnable:
                learn_body_settings = learnable_body_dict[body.name()]
            else:
                learn_body_settings = LearnableBodySettings()

            body_parameter = [
                Parameter(
                    body_param_tensor[0],
                    requires_grad=learn_body_settings.inertia_mass),
                Parameter(
                    body_param_tensor[1:4],
                    requires_grad=learn_body_settings.inertia_com),
                Parameter(
                    body_param_tensor[4:],
                    requires_grad=learn_body_settings.inertia_moments_products),
            ]
            self.body_parameters.extend(body_parameter)

        self.learnable_body_idx = learnable_body_idx

    # noinspection PyUnresolvedReferences
    @staticmethod
    def extract_body_parameters_and_variables(
            plant: MultibodyPlant_[Expression],
            model_ids: List[ModelInstanceIndex],
            context: Context,
            learnable_body_dict: Dict[str, LearnableBodySettings] = {}
    ) -> Tuple[List[Tensor], np.ndarray, List[DrakeBody]]:
        """Generates parameterization and symbolic variables for all bodies.

        For a multibody plant, finds all bodies that should have inertial
        properties; extracts the current values as an initial condition for
        ``theta``-format learnable parameters, and sets new symbolic versions of
        these variables.

        Args:
            plant: Symbolic plant from which to extract parameterization.
            model_ids: List of models in plant.
            context: Plant's symbolic context.
            learnable_body_dict: Dict of bodies and their learnable parameter
                settings.

        Returns:
            (n_bodies, 10) ``theta`` parameters initial conditions.
            (n_learnable_bodies, 10) symbolic inertial variables for any
                learnable bodies.
            (n_bodies,) list of inertial bodies.
            (n_learnable_bodies,) list of learnable body indices.
        """
        inertial_bodies, inertial_body_ids = \
            drake_utils.get_all_inertial_bodies(plant, model_ids)

        body_parameter_list = []
        body_variable_list = []
        learnable_indices = []

        bodies_and_ids = zip(inertial_bodies, inertial_body_ids)
        for i, body_and_id in enumerate(bodies_and_ids):
            body, body_id = body_and_id

            # get original values
            body_parameter_list.append(
                InertialParameterConverter.drake_to_theta(
                    body.CalcSpatialInertiaInBodyFrame(context)))

            # Don't parameterize any bodies whose inertial parameters are not
            # learnable.
            inertia_learnable, _, _ = are_body_properties_learnable(
                body.name(), learnable_body_dict)
            if not inertia_learnable:
                continue

            learnable_indices.append(i)

            mass = Variable(f'{body_id}_m', Variable.Type.CONTINUOUS)
            p_BoBcm_B = MakeVectorVariable(CENTER_OF_MASS_DOF, f'{body_id}_com',
                                           Variable.Type.CONTINUOUS)
            I_BBcm_B = MakeVectorVariable(INERTIA_TENSOR_DOF, f'{body_id}_I',
                                          Variable.Type.CONTINUOUS)

            body_spatial_inertia = \
                SpatialInertia_[Expression].MakeFromCentralInertia(
                    mass=mass, p_PScm_E=p_BoBcm_B,
                    I_SScm_E=RotationalInertia_[Expression](*I_BBcm_B))

            body.SetMass(context, mass)
            body.SetSpatialInertiaInBodyFrame(context, body_spatial_inertia)
            body_variable_list.append(np.hstack((mass, p_BoBcm_B, I_BBcm_B)))

        body_variables = np.vstack(body_variable_list)
        # pylint: disable=E1103
        return body_parameter_list, body_variables, inertial_bodies, \
            learnable_indices

    def pi_cm(self, just_learnables: bool = False) -> Tensor:
        """Returns inertial parameters in human-understandable ``pi_cm``
        -format."""
        indices = self.learnable_body_idx if just_learnables else \
            range(len(self.body_parameters)//3)

        inertial_parameters = []
        for idx in indices:
            inertial_parameters.append(torch.hstack(
                (self.body_parameters[3*idx],
                 self.body_parameters[3*idx+1],
                 self.body_parameters[3*idx+2])))

        return InertialParameterConverter.theta_to_pi_cm(
            torch.stack(inertial_parameters))

    def forward(self, q: Tensor, v: Tensor, u: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluates Lagrangian dynamics terms at given state and input.

        Args:
            q: (\*, n_q) configuration batch.
            v: (\*, n_v) velocity batch.
            u: (\*, n_u) input batch.

        Returns:
            (\*, n_v, n_v) mass matrix batch M(q)
            (\*, n_v) Lagrangian contact-free acceleration inv(M(q)) F(q)
        """
        # Pylint bug: cannot recognize instance attributes as Callable.
        # pylint: disable=not-callable
        assert self.mass_matrix is not None
        assert self.lagrangian_forces is not None
        learnable_inertia = \
            InertialParameterConverter.pi_cm_to_drake_spatial_inertia_vector(
            self.pi_cm(just_learnables=True))
        learnable_inertia = learnable_inertia.expand(
            q.shape[:-1] + learnable_inertia.shape)
        M = self.mass_matrix(q, learnable_inertia)
        non_contact_acceleration = torch.linalg.solve(
            M, self.lagrangian_forces(q, v, u, learnable_inertia))
        return M, non_contact_acceleration


ConfigurationCallback = Callable[[Tensor], Tensor]


def make_configuration_callback(expression: np.ndarray, q: np.ndarray) -> \
        Callable[[Tensor], Tensor]:
    """Converts drake symbolic expression to pytorch function via
    ``drake_pytorch``."""
    print(f'\nMAKING CONFIGURATION DRAKE PYTORCH EXPRESSION\n')
    return cast(
        Callable[[Tensor], Tensor],
        drake_pytorch.sym_to_pytorch(
            expression, q, simplify_computation=DEFAULT_SIMPLIFIER)[0])


def check_enabled(fn):
    def wrapped_fn(self, *args, **kwargs):
        if self.enabled:
            return fn(self, *args, **kwargs)
    return wrapped_fn
class HookGradientVisualizer:
    def __init__(self, module, vis_gradient=False):
        self.grad = None
        self.count_fw = 0
        self.count_bw = 0
        self.enabled = vis_gradient
        if self.enabled:
            print(f'HookGradientVisualizer is enabled. ')
        # self.loss_grad_to_vis = ['pred', 'comp', 'pen', 'diss']
        self.loss_grad_to_vis = ['comp']

        self.loss_grad_to_vis += ['all']
        self.cycle_bw = len(self.loss_grad_to_vis)

        self.module = module
    
    @check_enabled
    def manage_vis(self, loss_pred, loss_comp, loss_pen, loss_diss, 
                   w_pred, w_comp, w_pen, w_diss):
        for loss_name in self.loss_grad_to_vis:
            if loss_name == 'pred':
                loss_to_vis = w_pred * loss_pred
            elif loss_name == 'comp':
                loss_to_vis = w_comp * loss_comp
            elif loss_name == 'pen':
                loss_to_vis = w_pen * loss_pen
            elif loss_name == 'diss':
                loss_to_vis = w_diss * loss_diss
            elif loss_name == 'all':
                continue

            loss_to_vis = loss_to_vis.mean()
            print(f'{loss_name=}')
            loss_to_vis.backward(retain_graph=True)
            self.module.zero_grad()

    @check_enabled
    def record_geometries(self, i_contact, p_Ao, p_Bo, p_Ac, p_Bc, p_As, p_Bs, R_BW):
        # It will be called n_body_pair times.  
        # For robot exp, the first pair is sphere-object. The second pair is ground-object.
        if i_contact == 0:
            self.p_Ao = p_Ao.detach().numpy() # (n_frame, n_body_pair, 3)
            self.p_Bo = p_Bo.detach().numpy() # (n_frame, n_body_pair, 3)
            self.p_Ac = p_Ac.detach().numpy() # (n_frame, n_contact_at_this_pair, 3)
            self.p_Bc = p_Bc.detach().numpy() # (n_frame, n_contact_at_this_pair, 3)
            self.p_As = p_As.copy() # (n, n, 3)
            self.p_Bs = p_Bs.copy() # (n_samples, n_frame, 1, 3)
            self.R_BW = R_BW.detach().clone()   # (n_frame, 3, 3)

            self.count_fw += 1
            print(f'Forward pass {self.count_fw}')

            # In SupervisedLearningExperiment.train() in experiment.py, 
            # before getting into the training loops, there are one training epoch and one validation epoch.
            # One epoch only has one iteration. 
            # Therefore, self.count_fw=3 is the first training pass that we want to track. 
            # In the training loop, one training step and one eval step interleaved.
            # We only track the training step. Therefore, we take mod 2. 
            self.epoch = (self.count_fw - 3) // 2 # -1 // 2 == -1
            self.training_pass = (self.count_fw - 3) % 2 == 0
        else:
            p_Ac = p_Ac.detach().numpy()
            p_Bc = p_Bc.detach().numpy()
            self.p_Ac = np.concatenate((self.p_Ac, p_Ac), axis=-2)
            self.p_Bc = np.concatenate((self.p_Bc, p_Bc), axis=-2)
            self.p_As = [self.p_As, p_As]

    @check_enabled
    def record_dv(self, dv, dv_pred):
        self.dv = dv.detach().clone()
        self.dv_pred = dv_pred.detach().clone()

        self.dv_rot_axis_B, self.dv_rot_norm, self.dv_trans_axis_W, self.dv_trans_norm = self._process_dv(self.dv)
        self.dv_pred_rot_axis_B, self.dv_pred_rot_norm, self.dv_pred_trans_axis_W, self.dv_pred_trans_norm = \
            self._process_dv(self.dv_pred)
        
        self.dv_rot_axis_W = pbmm(self.dv_rot_axis_B.unsqueeze(-2), self.R_BW).squeeze(-2)
        self.dv_pred_rot_axis_W = pbmm(self.dv_pred_rot_axis_B.unsqueeze(-2), self.R_BW).squeeze(-2)
        
    def _process_dv(self, dv):
        dv_rot = dv[:, 0, :3]   # (*, 3)
        dv_rot_norm =  torch.norm(dv_rot, dim=-1, keepdim=True)
        dv_rot_axis = dv_rot / dv_rot_norm
        dv_trans = dv[:, 0, 3:] # (*, 3)
        dv_trans_norm = torch.norm(dv_trans, dim=-1, keepdim=True)
        dv_trans_axis = dv_trans / dv_trans_norm
        return dv_rot_axis, dv_rot_norm, dv_trans_axis, dv_trans_norm

    @check_enabled
    def record_impulses(self, impulses_by_contacts):
        self.impulses_by_contacts = impulses_by_contacts.detach().clone().squeeze(-1)   # (*, n_contacts, 3)
        self.impulses_by_contacts_norm = torch.norm(self.impulses_by_contacts, dim=-1, keepdim=True)
        self.impulses_by_contacts_normed = self.impulses_by_contacts / self.impulses_by_contacts_norm * 0.05

        self.impulses_by_contacts = self.impulses_by_contacts.numpy()
        self.impulses_by_contacts_norm = self.impulses_by_contacts_norm.numpy()
        self.impulses_by_contacts_normed = self.impulses_by_contacts_normed.numpy()

    @check_enabled
    def record(self, name, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().clone()
            self.__setattr__(name, x)
            self.__setattr__(name+'_normed', x / torch.norm(x, dim=-1, keepdim=True))
        elif isinstance(x, np.ndarray):
            self.__setattr__(name, x.copy())
        else:
            raise ValueError(f'Unsupported type: {type(x)}')

    @check_enabled
    def hook_check_grad(self, name, grad):
        ### Should only be enabled if only comp loss is backproped. 
        print(f'{name}_grad: {grad[0]}')
        grad_normed = grad / torch.norm(grad, dim=-1, keepdim=True)
        if name == 'p_BoBc_B':
            if torch.allclose(grad_normed, self.directions_A_to_B_in_B_normed, atol=1e-3):
                print("p_BoBc_B's grad is d_AB_B_normed, "+ \
                      "which is expected for complementarity loss. ")
            else:
                print("p_BoBc_B's grad is not d_AB_B_normed! " + \
                      "Check if it is because other losses are also backproped. ")
                breakpoint()
        elif name == 'p_BoBc_A':
            if torch.allclose(grad_normed, self.directions_A_to_B_in_A_normed, atol=1e-3):
                print("p_BoBc_A's grad is d_AB_A_normed, "+ \
                      "which is expected for complementarity loss. ")
            else:
                print("p_BoBc_A's grad is not d_AB_A_normed!" + \
                      "Check if it is because other losses are also backproped. ")
                breakpoint()
        elif name == 'p_AcBc_A':
            if torch.allclose(grad_normed, self.directions_A_to_B_in_A_normed, atol=1e-3):
                print("p_AcBc_A's grad is d_AB_A_normed, "+ \
                      "which is expected for complementarity loss. ")
            else:
                print("p_AcBc_A's grad is not d_AB_A_normed!" + \
                      "Check if it is because other losses are also backproped. ")
                breakpoint()

    @check_enabled
    def hook_grad_plane_and_object(self, p_BiBc_B_grad):
        print('hook_grad_plane_and_object')
        if self.training_pass and self.epoch >= 0:
            self.p_BiBc_B_grad_plane = p_BiBc_B_grad.detach().clone()   # (batch_size, n_c=5, 3)
            self.p_BiBc_W_grad_plane = pbmm(self.p_BiBc_B_grad_plane, self.R_BW).detach().numpy()
            # self.p_BiBc_W_grad.shape = (batch_size, n_c, 3)
            self.p_BiBc_W_grad_plane_norm = np.linalg.norm(self.p_BiBc_W_grad_plane, axis=2, keepdims=True)
            self.p_BiBc_W_grad_plane_normalized = self.p_BiBc_W_grad_plane / self.p_BiBc_W_grad_plane_norm * 0.05
            self.p_BiBc_W_grad_plane = - self.p_BiBc_W_grad_plane
            self.p_BiBc_W_grad_plane_normalized = - self.p_BiBc_W_grad_plane_normalized

    @check_enabled
    def hook_grad_sphere_and_object(self, p_BiBc_B_grad):
        print('hook_grad_sphere_and_object')
        self.count_bw += 1
        print(f'Backward pass {self.count_bw}')
        if self.training_pass and self.epoch >= 0:
            # MultibodyLearnableSystem.contactnets_loss() in multibody_learnable_system.py,
            # four backward passes are called for each loss terms before the full-loss backward pass.
            # The training epoch before the training loops does not use optimizer, 
            # thus one fewer backward pass per forward pass in that epoch. 
            self.loss_name = self.loss_grad_to_vis[self.count_bw % self.cycle_bw]
            print(f"{self.loss_name=}")
            
            self.p_BiBc_B_grad = p_BiBc_B_grad.detach().clone()
            self.p_BiBc_B_grad = self.p_BiBc_B_grad.unsqueeze(1)
            self.p_BiBc_W_grad = pbmm(self.p_BiBc_B_grad, self.R_BW).detach().numpy()
            self.directions_B_W = pbmm(self.directions_B[:,None], self.R_BW).detach().numpy() * 0.05
            # self.p_BiBc_W_grad.shape = (batch_size, n_c=1, 3)
            self.p_BiBc_W_grad_norm = np.linalg.norm(self.p_BiBc_W_grad, axis=2, keepdims=True)
            self.p_BiBc_W_grad_normalized = self.p_BiBc_W_grad / self.p_BiBc_W_grad_norm * 0.05
            self.p_BiBc_W_grad = - self.p_BiBc_W_grad
            self.p_BiBc_W_grad_normalized = - self.p_BiBc_W_grad_normalized

            self.p_BiBc_W_grad = np.concatenate([self.p_BiBc_W_grad, self.p_BiBc_W_grad_plane], axis=-2)
            self.p_BiBc_W_grad_normalized = np.concatenate([self.p_BiBc_W_grad_normalized, 
                                                            self.p_BiBc_W_grad_plane_normalized], axis=-2)
            self.p_BiBc_W_grad_norm = np.linalg.norm(self.p_BiBc_W_grad, axis=2, keepdims=True)
            self._visualize()
            

    def _visualize_single_view(self, ax, i, azim=0):
        # Plot the sphere
        x = self.p_As[0][..., 0] + self.p_Ao[i, 0, 0]
        y = self.p_As[0][..., 1] + self.p_Ao[i, 0, 1]
        z = self.p_As[0][..., 2] + self.p_Ao[i, 0, 2]
        ax.plot_surface(x, y, z, color='b', alpha=0.2)

        # Plot the plane (if the plane origin is not (0,0,0), 
        # the code here and p_AoAs_W calculation in ContactTerms.forward() needs change)
        x = self.p_As[1][..., 0] + self.p_Bo[i, 0, 0] + self.p_Ao[i, 1, 0]
        y = self.p_As[1][..., 1] + self.p_Bo[i, 0, 1] + self.p_Ao[i, 1, 1]
        z = self.p_As[1][..., 2] + self.p_Ao[i, 1, 2]
        ax.plot_surface(x, y, z, color='b', alpha=0.2)
        
        # Plot the object
        x_obj = self.p_Bs[:, i, :, 0] + self.p_Bo[i, 0, 0]
        y_obj = self.p_Bs[:, i, :, 1] + self.p_Bo[i, 0, 1]
        z_obj = self.p_Bs[:, i, :, 2] + self.p_Bo[i, 0, 2]
        ax.scatter(x_obj, y_obj, z_obj, color='r', alpha=0.2, s=2)

        # Plot the sphere origin
        ax.scatter(self.p_Ao[i, 0, 0], self.p_Ao[i, 0, 1], self.p_Ao[i, 0, 2], color='blue', s=10)
        # Plot the obj origin
        ax.scatter(self.p_Bo[i, 0, 0], self.p_Bo[i, 0, 1], self.p_Bo[i, 0, 2], color='red', s=10)
        # Plot the sphere witness contact point
        ax.scatter(self.p_Ac[i, 0, 0], self.p_Ac[i, 0, 1], self.p_Ac[i, 0, 2], color='blue', marker='x')
        # Plot the obj witness contact point
        ax.scatter(self.p_Bc[i, 0, 0], self.p_Bc[i, 0, 1], self.p_Bc[i, 0, 2], color='red', marker='x')

        # Plot the obj query direction
        # ax.quiver(self.p_Bo[i, 0, 0], self.p_Bo[i, 0, 1], self.p_Bo[i, 0, 2],
        #             self.directions_B_W[i, 0, 0], 
        #             self.directions_B_W[i, 0, 1], 
        #             self.directions_B_W[i, 0, 2], color='blue', linewidth=0.5)
        
        # Plot the angular acceleration induced by contact impulses
        # 1/30 is dt. *5 is only for visualization.
        ax.quiver(self.p_Bo[i, 0, 0], self.p_Bo[i, 0, 1], self.p_Bo[i, 0, 2],
                    self.dv_pred_rot_axis_W[i, 0] * self.dv_pred_rot_norm[i, 0] / 30 * 5, # 30Hz
                    self.dv_pred_rot_axis_W[i, 1] * self.dv_pred_rot_norm[i, 0] / 30 * 5,
                    self.dv_pred_rot_axis_W[i, 2] * self.dv_pred_rot_norm[i, 0] / 30 * 5, color='blue', label='dv_pred_r')
        # Plot the normalized value at 0.05 scale. 
        ax.quiver(self.p_Bo[i, 0, 0], self.p_Bo[i, 0, 1], self.p_Bo[i, 0, 2],
                    self.dv_pred_rot_axis_W[i, 0] * 0.05,
                    self.dv_pred_rot_axis_W[i, 1] * 0.05,
                    self.dv_pred_rot_axis_W[i, 2] * 0.05, color='blue', linewidth=0.5)
        
        # Plot the linear acceleration induced by contact impulses
        ax.quiver(self.p_Bo[i, 0, 0], self.p_Bo[i, 0, 1], self.p_Bo[i, 0, 2],
                  self.dv_pred_trans_axis_W[i, 0] * self.dv_pred_trans_norm[i, 0] / 30 * 5,
                  self.dv_pred_trans_axis_W[i, 1] * self.dv_pred_trans_norm[i, 0] / 30 * 5,
                  self.dv_pred_trans_axis_W[i, 2] * self.dv_pred_trans_norm[i, 0] / 30 * 5, color='purple', label='dv_pred_t')
        ax.quiver(self.p_Bo[i, 0, 0], self.p_Bo[i, 0, 1], self.p_Bo[i, 0, 2],
                  self.dv_pred_trans_axis_W[i, 0] * 0.05,
                  self.dv_pred_trans_axis_W[i, 1] * 0.05,
                  self.dv_pred_trans_axis_W[i, 2] * 0.05, color='purple', linewidth=0.5)
                  
        # Plot the angular acceleration observed in data
        ax.quiver(self.p_Bo[i, 0, 0], self.p_Bo[i, 0, 1], self.p_Bo[i, 0, 2],
                    self.dv_rot_axis_W[i, 0] * self.dv_rot_norm[i, 0] / 30 * 5, # 30Hz
                    self.dv_rot_axis_W[i, 1] * self.dv_rot_norm[i, 0] / 30 * 5,
                    self.dv_rot_axis_W[i, 2] * self.dv_rot_norm[i, 0] / 30 * 5, color='blue', label='dv_r', linestyle=':')
        ax.quiver(self.p_Bo[i, 0, 0], self.p_Bo[i, 0, 1], self.p_Bo[i, 0, 2],
                    self.dv_rot_axis_W[i, 0] * 0.05,
                    self.dv_rot_axis_W[i, 1] * 0.05,
                    self.dv_rot_axis_W[i, 2] * 0.05, color='blue', linewidth=0.5, linestyle=':')
        
        # Plot the linear acceleration observed in data
        ax.quiver(self.p_Bo[i, 0, 0], self.p_Bo[i, 0, 1], self.p_Bo[i, 0, 2],
                  self.dv_trans_axis_W[i, 0] * self.dv_trans_norm[i, 0] / 30 * 5,
                  self.dv_trans_axis_W[i, 1] * self.dv_trans_norm[i, 0] / 30 * 5,
                  self.dv_trans_axis_W[i, 2] * self.dv_trans_norm[i, 0] / 30 * 5, color='purple', label='dv_t', linestyle=':')
        ax.quiver(self.p_Bo[i, 0, 0], self.p_Bo[i, 0, 1], self.p_Bo[i, 0, 2],
                  self.dv_trans_axis_W[i, 0] * 0.05,
                  self.dv_trans_axis_W[i, 1] * 0.05,
                  self.dv_trans_axis_W[i, 2] * 0.05, color='purple', linewidth=0.5, linestyle=':')
        
        # Plot the gradient descent direction of the obj contact point wrt the object origin in world frame
        ax.quiver(self.p_Bc[i, :, 0], self.p_Bc[i, :, 1], self.p_Bc[i, :, 2],
                    self.p_BiBc_W_grad_normalized[i, :, 0], 
                    self.p_BiBc_W_grad_normalized[i, :, 1], 
                    self.p_BiBc_W_grad_normalized[i, :, 2], color='red', linewidth=0.5)
        
        ax.quiver(self.p_Bc[i, :, 0], self.p_Bc[i, :, 1], self.p_Bc[i, :, 2], 
                    self.p_BiBc_W_grad[i, :, 0], 
                    self.p_BiBc_W_grad[i, :, 1], 
                    self.p_BiBc_W_grad[i, :, 2], color='red', label='gradients')

        # Plot the impulses estimated in the inner loop convex optimization
        ax.quiver(self.p_Bc[i, :, 0], self.p_Bc[i, :, 1], self.p_Bc[i, :, 2],
                    self.impulses_by_contacts_normed[i, :, 0], 
                    self.impulses_by_contacts_normed[i, :, 1], 
                    self.impulses_by_contacts_normed[i, :, 2], color='green', linewidth=0.5)
        
        ax.quiver(self.p_Bc[i, :, 0], self.p_Bc[i, :, 1], self.p_Bc[i, :, 2], 
                    self.impulses_by_contacts[i, :, 0], 
                    self.impulses_by_contacts[i, :, 1], 
                    self.impulses_by_contacts[i, :, 2], color='green', label='impulses')
        
        if azim == 0:
            ax.legend()
        ax.view_init(elev=0, azim=azim)
        ax.set_box_aspect([np.ptp(arr) for arr in \
        [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        # use :.2e format for array self.p_BiBc_W_grad_norm[i, :, 0] which has multiple elements
        formatted_grad_array = np.array2string(self.p_BiBc_W_grad_norm[i, :, 0], 
                                formatter={'float_kind': lambda x: '{:.2e}'.format(x)},
                                separator=', ')

        formatted_impulses_array = np.array2string(self.impulses_by_contacts_norm[i, :, 0],
                                                   formatter={'float_kind': lambda x: '{:.2e}'.format(x)},
                                                   separator=', ')
        title_text = f'frame {i}, ||grad||: {formatted_grad_array}, \n ||imps||: {formatted_impulses_array}\n' + \
        f'||dv_pred_r||: {self.dv_pred_rot_norm[i, 0]:.2e}, ||dv_pred_t||: {self.dv_pred_trans_norm[i, 0]:.2e}, ' + \
        f'||dv_r||: {self.dv_rot_norm[i, 0]:.2e}, ||dv_t||: {self.dv_trans_norm[i, 0]:.2e}'
        
        return title_text

    def _visualize(self):
        video_output_file = op.join('/mnt/data0/minghz/repos/bundlenets/dair_pll', f'contact_geometry_{self.loss_name}17s0r_{self.epoch:04d}.mp4')
        print(f'Saving video to {video_output_file}')
        with TemporaryDirectory(prefix="sdf-slice-") as tmpdir:
            print(f'Storing temporary files at {tmpdir}')
            for i in tqdm(range(self.p_Ao.shape[0])):
                fig = plt.figure(figsize=(10, 5))
                ax = fig.add_subplot(121, projection='3d')
                self._visualize_single_view(ax, i, azim=0)

                ax = fig.add_subplot(122, projection='3d')
                title_text = self._visualize_single_view(ax, i, azim=90) # looking from the right

                title_text = f'{self.loss_name}, epoch {self.epoch}, {title_text}'
                fig.suptitle(title_text)

            # plt.show(block=False)
            # breakpoint()
                if video_output_file is not None:
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.savefig(op.join(tmpdir, f'{i:07d}.png'))

            if video_output_file is not None:
                os.system(f'ffmpeg -y -r 30 -i {tmpdir}/%07d.png -vcodec ' + \
                        f'libx264 -preset slow -crf 18 {video_output_file}')
                print(f'Saved slice video to {video_output_file}.')


class ContactTerms(Module):
    """Container class for contact-related dynamics terms.

    Derives batched pytorch callback functions for collision geometry
    position and velocity kinematics from a
    :class:`~dair_pll.drake_utils.MultibodyPlantDiagram`.
    """
    geometry_rotations: Optional[ConfigurationCallback]
    geometry_translations: Optional[ConfigurationCallback]
    geometry_spatial_jacobians: Optional[ConfigurationCallback]
    geometries: ModuleList
    friction_param_list: ParameterList
    friction_params: Tensor
    collision_candidates: Tensor

    def __init__(self, plant_diagram: MultibodyPlantDiagram,
                 represent_geometry_as: str = 'box',
                 learnable_body_dict: Dict[str, LearnableBodySettings] = {},
                 vis_hook: HookGradientVisualizer = None,
                 ) -> None:
        """Inits :py:class:`ContactTerms` with prescribed kinematics and
        geometries.

        phi(q) and J(q) are calculated implicitly from kinematics and ``n_g ==
        len(geometries)`` collision geometries C.

        Args:
            plant_diagram: Drake MultibodyPlant diagram to extract terms from.
            represent_geometry_as: How to represent the geometry of any
              learnable bodies (box/mesh/polygon).  By default, any ``Plane``
              objects are not considered learnable -- only boxes or meshes.
        """
        # pylint: disable=too-many-locals
        super().__init__()
        plant, context, q, v = init_symbolic_plant_context_and_state(
            plant_diagram)
        inspector = plant_diagram.scene_graph.model_inspector()

        collision_geometry_set = plant_diagram.collision_geometry_set
        geometry_ids = collision_geometry_set.ids
        coulomb_frictions = collision_geometry_set.frictions

        # sweep over collision elements
        geometries, rotations, translations, drake_spatial_jacobians = \
            ContactTerms.extract_geometries_and_kinematics(
                plant, inspector, geometry_ids, context, represent_geometry_as,
                learnable_body_dict=learnable_body_dict)

        collision_candidates = []
        # If training, only consider collisions between a body with learnable
        # geometry and anything else.
        if self.training:
            for candidate in collision_geometry_set.collision_candidates:
                if geometries[candidate[0]].learnable or \
                    geometries[candidate[1]].learnable:
                    collision_candidates.append(candidate)
        else:
            collision_candidates = collision_geometry_set.collision_candidates

        for geometry_index, geometry_pair in enumerate(collision_candidates):
            if geometries[geometry_pair[0]] > geometries[geometry_pair[1]]:
                collision_candidates[geometry_index] = (geometry_pair[1],
                                                        geometry_pair[0])

            # Ensure the learnable body is listed second.
            # TODO: Make this more generalizable.  For now, this is necessary
            # because ``forward`` returns the witness points on the learnable
            # geometry, as distinct from the witness points on the nonlearnable
            # geometries.
            assert geometries[
                collision_candidates[geometry_index][1]].learnable, \
                f'Collision pair {collision_candidates[geometry_index]} ' + \
                f'requires learnable object to be listed second.'

        self.geometry_rotations = make_configuration_callback(
            np.stack(rotations), q)

        self.geometry_translations = make_configuration_callback(
            np.stack(translations), q)

        drake_velocity_jacobian = Jacobian(plant.GetVelocities(context), v)
        self.geometry_spatial_jacobians = make_configuration_callback(
            np.stack([
                jacobian @ drake_velocity_jacobian
                for jacobian in drake_spatial_jacobians
            ]), q)

        self.geometries = ModuleList(geometries)

        self.friction_param_list = ParameterList()
        for idx, friction in enumerate(coulomb_frictions):
            body = drake_utils.get_body_from_geometry_id(
                plant, inspector, geometry_ids[idx])
            _, _, friction_learnable = are_body_properties_learnable(
                body.name(), learnable_body_dict)
            self.friction_param_list.append(Parameter(
                torch.tensor([friction.static_friction()]),
                requires_grad=friction_learnable)
            )

        self.collision_candidates = torch.tensor(
            collision_candidates).t().long()
        
        self.vis_hook = vis_hook

    def get_friction_coefficients(self) -> Tensor:
        """From the stored :py:attr:`friction_param_list`, compute the friction
        coefficient as its absolute value."""
        return torch.abs(torch.hstack([
            param for param in self.friction_param_list]))

    # noinspection PyUnresolvedReferences
    @staticmethod
    def extract_geometries_and_kinematics(
        plant: MultibodyPlant_[Expression], inspector: SceneGraphInspector,
        geometry_ids: List[GeometryId], context: Context,
        represent_geometry_as: str,
        learnable_body_dict: Dict[str, LearnableBodySettings] = {}
    ) -> Tuple[List[CollisionGeometry], List[np.ndarray], List[np.ndarray],
               List[np.ndarray]]:
        """Extracts modules and kinematics of list of geometries G.

        Args:
            plant: Multibody plant from which terms are extracted.
            inspector: Scene graph inspector associated with plant.
            geometry_ids: List of geometries to model.
            context: Plant's context with symbolic state.
            represent_geometry_as: How to represent learnable geometries.

        Returns:
            List of :py:class:`CollisionGeometry` models with one-to-one
              correspondence with provided geometries.
            List[(3,3)] of corresponding rotation matrices R_WG
            List[(3,)] of corresponding geometry frame origins p_WoGo_W
            List[(6,n_v)] of geometry spatial jacobians w.r.t. drake velocity
              coordinates, J(v_drake)_V_WG_W
        """
        world_frame = plant.world_frame()
        geometries = []
        rotations = []
        translations = []
        drake_spatial_jacobians = []

        for geometry_id in geometry_ids:
            geometry_pose = inspector.GetPoseInFrame(
                geometry_id).cast[Expression]()

            body = plant.GetBodyFromFrameId(inspector.GetFrameId(geometry_id))

            _, learnable_geometry, _ = are_body_properties_learnable(
                body.name(), learnable_body_dict)

            geometry_frame = body.body_frame()

            geometry_transform = geometry_frame.CalcPoseInWorld(
                context) @ geometry_pose

            rotations.append(geometry_transform.rotation().matrix())

            translations.append(geometry_transform.translation())

            drake_spatial_jacobian = plant.CalcJacobianSpatialVelocity(
                context=context,
                with_respect_to=JacobianWrtVariable.kV,
                frame_B=geometry_frame,
                p_BoBp_B=geometry_pose.translation().reshape(3, 1),
                frame_A=world_frame,
                frame_E=world_frame)
            drake_spatial_jacobians.append(drake_spatial_jacobian)

            geometries.append(
                PydrakeToCollisionGeometryFactory.convert(
                    inspector.GetShape(geometry_id), represent_geometry_as,
                    learnable_geometry, body.name()))

        return geometries, rotations, translations, drake_spatial_jacobians

    @staticmethod
    def assemble_velocity_jacobian(R_CW, Jv_V_WC_W, p_CoCc_C):
        """Helper method to generate velocity jacobian from contact information.

        Args:
            R_CW: (\*, n_c, 3, 3) Rotation of world frame w.r.t. geometry frame.
            Jv_V_WC_W: (\*, 1, 6, n_v) Geometry spatial velocity Jacobian.
            p_CoCc_C: (\*, n_c, 3) Geometry-frame contact points.

        Returns:
            (\*, n_c, 3, n_v) World-frame contact point translational velocity
            Jacobian.
        """
        p_CoCc_W = pbmm(p_CoCc_C.unsqueeze(-2), R_CW).squeeze(-2)
        Jv_v_WCc_W = pbmm(spatial_to_point_jacobian(p_CoCc_W), Jv_V_WC_W)
        # Jv_v_WCc_W: (\*, n_c, 3, n_v) contact point translational velocity 
        # Jacobian wrt system velocity in world frame.
        return Jv_v_WCc_W

    @staticmethod
    def relative_velocity_to_contact_jacobian(Jv_v_W_BcAc_F: Tensor,
                                              mu: Tensor) -> Tensor:
        """Helper method to reorder contact Jacobian columns.

        Args:
            Jv_v_W_BcAc_F: (\*, n_collisions, 3, n_v) collection of
            contact-frame relative velocity Jacobians.
            mu: (n_collisions,) list of

        Returns:
            (\*, 3 * n_collisions, n_v) contact jacobian J(q) in [J_n; mu * J_t]
            ordering.
        """
        # Tuple of (*, n_collisions, n_v)
        J_x, J_y, J_z = deal(Jv_v_W_BcAc_F, -2)

        J_n = J_z

        # Reshape (*, n_collisions, 2 * n_v) -> (*, 2 * n_collisions, n_v)
        # pylint: disable=E1103
        mu_shape = torch.Size((1,) * (J_x.dim() - 2) + mu.shape + (1,))
        friction_jacobian_shape = J_x.shape[:-2] + (-1, J_x.shape[-1])
        J_t = (mu.reshape(mu_shape) * torch.cat((J_x, J_y), dim=-1)) \
            .reshape(friction_jacobian_shape)
        return torch.cat((J_n, J_t), dim=-2)

    def forward(self, q: Tensor
                ) -> Tuple[Tensor, Tensor, Tensor, List, List, List]:
        """Evaluates Lagrangian dynamics terms at given state and input.

        Uses :py:class:`GeometryCollider` and kinematics to construct signed
        distance phi(q) and the corresponding Jacobian J(q).

        phi(q) and J(q) are calculated implicitly from kinematics and collision
        geometries.

        Args:
            q: (\*, n_q) configuration batch.

        Returns:
            (\*, n_collisions) signed distance phi(q).
            (\*, 3 * n_collisions, n_v) contact Jacobian J(q).
            (\*, n_collisions, 3) contact point in body B frame, where body B is
                the second body in each collision pair.  From the ``__init__``,
                this body is always a learnable body.
        """
        # Pylint bug: cannot recognize instance attributes as Callable.
        # pylint: disable=too-many-locals,not-callable
        assert self.geometry_rotations is not None
        assert self.geometry_translations is not None
        assert self.geometry_spatial_jacobians is not None
        R_WC = self.geometry_rotations(q)
        p_WoCo_W = self.geometry_translations(q)
        Jv_V_WC_W = self.geometry_spatial_jacobians(q)

        indices_a = self.collision_candidates[0, :]
        indices_b = self.collision_candidates[1, :]

        geometries_a = [
            cast(CollisionGeometry, self.geometries[element_index])
            for element_index in indices_a
        ]
        geometries_b = [
            cast(CollisionGeometry, self.geometries[element_index])
            for element_index in indices_b
        ]

        friction_coefficients = self.get_friction_coefficients()
        mu_a = friction_coefficients[indices_a]
        mu_b = friction_coefficients[indices_b]

        # combine friction coefficients as in Drake.
        mu = (2 * mu_a * mu_b) / (mu_a + mu_b)

        R_WA = R_WC[..., indices_a, :, :]
        R_AW = deal(R_WA.transpose(-1, -2), -3)
        R_WB = R_WC[..., indices_b, :, :]
        R_BW = deal(R_WB.transpose(-1, -2), -3)

        Jv_V_WA_W = deal(Jv_V_WC_W[..., indices_a, :, :], -3, keep_dim=True)
        Jv_V_WB_W = deal(Jv_V_WC_W[..., indices_b, :, :], -3, keep_dim=True)
        # For robot experiments:
        # Jv_V_WA_W: tuple of length n_ab_pairs, each (\*, 1, 6, n_v): jacobian of
        #   spatial velocity of body A w.r.t. system velocities in world frame
        # Jv_V_WB_W: tuple of length n_ab_pairs, each (\*, 1, 6, n_v): jacobian of
        #   spatial velocity of body B w.r.t. system velocities in world frame
        # Body A: sphere and ground
        # Body B: object
        # n_v = 13. Among the 13 system velocities, the last 6 are the rotational and linear
        #   velocities of body B. Rotation comes first, then translation.
        #   (See spatial_to_point_jacobian() in tensor_utils.py)
        #   (See dair_pll/assets/precomputed_vision_functions/mass_matrix_state_names.txt
        #   for the order of all states.)
        # The last 3x3 submatrix is always identity. 
        # If object rotation pose is identity, then the 6x6 submatrix is identity.
        # Meaning that the system object angular velocity is in the object frame, and the 
        # system object linear velocity is in the world frame.
        # print(f'{Jv_V_WB_W[0][0,0,:,-6:]=}')
        # print(f'{Jv_V_WB_W[1][0,0,:,-6:]=}')
        # breakpoint()

        # Interbody translation in A frame, shape (*, n_g, 3)
        p_AoBo_W = p_WoCo_W[..., indices_b, :] - p_WoCo_W[..., indices_a, :]
        p_AoBo_A = deal(pbmm(p_AoBo_W.unsqueeze(-2), R_WA).squeeze(-2), -2)

        p_WoAo_W = p_WoCo_W[..., indices_a, :]
        p_WoBo_W = p_WoCo_W[..., indices_b, :]
        # p_WoAo_W.shape == (batch_size, n_ab_pairs, 3)
        # p_WoAo_W[:,1].shape == (batch_size, 3), which is all zero, because geo_a is ground. 
        # R_WA[:,1].shape == (batch_size, 3, 3), which is identity matrix, because geo_a is ground.

        Jv_v_W_BcAc_F = []
        phi_list = []
        p_BiBc_B_list = []
        obj_pair_list = []
        R_FW_list = []
        mu_list = []

        # bundle all modules and kinematics into a tuple iterator
        a_b = zip(geometries_a, geometries_b, R_AW, R_BW, p_AoBo_A, Jv_V_WA_W,
                  Jv_V_WB_W, deal(mu))

        # iterate over body pairs (Ai, Bi)
        for i_pair, (geo_a, geo_b, R_AiW, R_BiW, p_AiBi_A, Jv_V_WAi_W, Jv_V_WBi_W, mu_i) \
            in enumerate(a_b):
            # relative rotation between Ai and Bi, (*, 3, 3)
            R_AiBi = pbmm(R_AiW, R_BiW.transpose(-1, -2))

            # collision result,
            # Tuple[(*, n_c), (*, n_c, 3, 3), (*, n_c, 3), (*, n_c, 3)]
            phi_i, R_AiF, p_AiAc_A, p_BiBc_B = GeometryCollider.collide(
                geo_a, geo_b, R_AiBi, p_AiBi_A, self.vis_hook)
            n_c = phi_i.shape[-1]

            # plot the geometries
            p_AiAc_W = pbmm(p_AiAc_A, R_AiW)
            p_WoAc_W = p_AiAc_W + p_WoAo_W[..., [i_pair], :]

            p_BiBc_W = pbmm(p_BiBc_B, R_BiW)
            p_WoBc_W = p_BiBc_W + p_WoBo_W[..., [i_pair], :]
            
            p_BoBs_B = geo_b.get_vertices(directions=np.ones([1,3]),sample_entire_mesh=True) 
            # shape (1, n_samples, 3), where n_samples comes from _GRID in deep_support_functions.py
            p_BoBs_B = p_BoBs_B.permute(1, 0, 2)[:,None] # shape (n_samples, 1, 1, 3)
            p_BoBs_W = pbmm(p_BoBs_B, R_BiW[None]) 
            # p_BoBs_W.shape (n_samples, 1, 1, 3) * (1, batch_size, 3, 3) = (n_samples, batch_size, 1, 3)
            p_BoBs_B = p_BoBs_B.expand_as(p_BoBs_W).detach().numpy()
            p_BoBs_W = p_BoBs_W.detach().numpy()

            if i_pair == 0:
                geom_sphere = geometries_a[0]
                sphere_radius = geom_sphere.get_radius().item()
                ### Generate the sphere's coordinates
                u = np.linspace(0, 2 * np.pi, 100)
                v = np.linspace(0, np.pi, 100)
                xs = np.outer(np.cos(u), np.sin(v)) * sphere_radius
                ys = np.outer(np.sin(u), np.sin(v)) * sphere_radius
                zs = np.outer(np.ones(u.shape), np.cos(v)) * sphere_radius
                p_AoAs_W = np.stack([xs, ys, zs], axis=-1) # shape (100, 100, 3)
            elif i_pair == 1:
                ### Find the xy range of the object B and use it as the plane A's extent for visualization
                # geom_plane = geometries_a[1]
                xyz_min = np.min(p_BoBs_W, axis=0) # shape (batch_size, 1, 3)
                xyz_min = np.min(xyz_min, axis=0)[0] # shape (1, 3)
                xyz_max = np.max(p_BoBs_W, axis=0) # shape (batch_size, 1, 3)
                xyz_max = np.max(xyz_max, axis=0)[0] # shape (1, 3)
                xs = np.linspace(xyz_min[0], xyz_max[0], 10)
                ys = np.linspace(xyz_min[1], xyz_max[1], 10)
                xs, ys = np.meshgrid(xs, ys)
                zs = np.zeros_like(xs)
                p_AoAs_W = np.stack([xs, ys, zs], axis=-1) # shape (10, 10, 3)

            self.vis_hook.record_geometries(i_pair, p_WoAo_W, p_WoBo_W, p_WoAc_W, p_WoBc_W, p_AoAs_W, p_BoBs_W, R_BiW)

            # contact frame rotation, (*, n_c, 3, 3)
            R_FW = pbmm(R_AiF.transpose(-1, -2), R_AiW.unsqueeze(-3))

            # contact point velocity jacobians, (*, n_c, 3, n_v)
            Jv_v_WAc_W = ContactTerms.assemble_velocity_jacobian(
                R_AiW.unsqueeze(-3), Jv_V_WAi_W, p_AiAc_A)
            Jv_v_WBc_W = ContactTerms.assemble_velocity_jacobian(
                R_BiW.unsqueeze(-3), Jv_V_WBi_W, p_BiBc_B)
            # How to read Jv_v_WAc_W: 
            # J[v: system velocities]_[v_WBc: relative velocity of point Bc wrt point W(world frame origin)]_
            # [W: velocity expressed in world frame]

            # contact relative velocity in contact frame jacobians wrt 
            # system velocities, (*, n_c, 3, n_v)
            Jv_v_W_BcAc_F.append(pbmm(R_FW, Jv_v_WBc_W - Jv_v_WAc_W))
            phi_list.append(phi_i)
            p_BiBc_B_list.append(p_BiBc_B)
            obj_pair_list.extend(n_c * [(geo_a.name, geo_b.name)])
            mu_list.extend(n_c * [mu_i])
            R_FW_list.extend([R_FW[..., i, :, :] for i in range(n_c)])

        # pylint: disable=E1103
        mu_repeated = torch.cat(
            [mu_i.repeat(phi_i.shape[-1]) for phi_i, mu_i in zip(phi_list, mu)])
        phi = torch.cat(phi_list, dim=-1)  # type: Tensor
        J = ContactTerms.relative_velocity_to_contact_jacobian(
            torch.cat(Jv_v_W_BcAc_F, dim=-3), mu_repeated)
        p_BiBc_B = torch.cat(p_BiBc_B_list, dim=-2)

        if torch.any(J.isnan()):
            pdb.set_trace()

        # Some size checking.
        n_lambda = phi.shape[-1]
        n_v = J.shape[-1]
        assert J.shape[-2:] == (3 * n_lambda, n_v)
        assert p_BiBc_B.shape[-2:] == (n_lambda, 3)
        assert len(obj_pair_list) == len(R_FW_list) == len(mu_list) == n_lambda

        return phi, J, p_BiBc_B, obj_pair_list, R_FW_list, mu_list

class MultibodyTerms(Module):
    """Derives and manages computation of terms of multibody dynamics with
    contact.

    Primarily
    """
    lagrangian_terms: LagrangianTerms
    contact_terms: ContactTerms
    geometry_body_assignment: Dict[str, List[int]]
    plant_diagram: MultibodyPlantDiagram
    urdfs: Dict[str, str]
    learnable_body_dict: Dict[str, LearnableBodySettings]
    pretrained_icnn_weights_filepath: str

    def scalars_and_meshes(
            self) -> Tuple[Dict[str, float], Dict[str, MeshSummary]]:
        """Generates summary statistics for inertial and geometric quantities."""
        scalars = {}
        meshes = {}
        _, all_body_ids = \
            drake_utils.get_all_inertial_bodies(
                self.plant_diagram.plant,
                self.plant_diagram.model_ids)

        friction_coefficients = self.contact_terms.get_friction_coefficients()

        for body_pi, body_id in zip(self.lagrangian_terms.pi_cm(), all_body_ids):

            body_scalars = InertialParameterConverter.pi_cm_to_scalars(body_pi)

            scalars.update({
                f'{body_id}_{scalar_name}': scalar
                for scalar_name, scalar in body_scalars.items()
            })

            for geometry_index in self.geometry_body_assignment[body_id]:
                # include geometry
                geometry = self.contact_terms.geometries[geometry_index]
                geometry_scalars = geometry.scalars()
                scalars.update({
                    f'{body_id}_{scalar_name}': scalar
                    for scalar_name, scalar in geometry_scalars.items()
                })

                # include friction
                scalars[f'{body_id}_mu'] = \
                    friction_coefficients[geometry_index].item()

                geometry_mesh = None
                if isinstance(geometry, DeepSupportConvex):
                    # print('>>>>>>>>>>>>', self.pretrained_icnn_weights_filepath)
                    # if self.pretrained_icnn_weights_filepath is not None:
                    #     print(f'Loading pretrained ICNN weight from ' \
                    #           + f'{self.pretrained_icnn_weights_filepath}')
                    #     geometry.load_weights(self.pretrained_icnn_weights_filepath)
                    geometry_mesh = extract_mesh_from_support_function(
                        geometry.network)

                elif isinstance(geometry, Polygon):
                    geometry_mesh = get_mesh_summary_from_polygon(geometry)

                if geometry_mesh is not None:
                    meshes[body_id] = geometry_mesh
                    vertices = geometry_mesh.vertices
                    diameters = vertices.max(dim=0).values - vertices.min(
                        dim=0).values
                    center = vertices.min(dim=0).values + diameters / 2
                    scalars.update({
                        f'{body_id}_diameter_{axis}': value.item()
                        for axis, value in zip(['x', 'y', 'z'], diameters)
                    })
                    scalars.update({
                        f'{body_id}_center_{axis}': value.item()
                        for axis, value in zip(['x', 'y', 'z'], center)
                    })
                    # if self.pretrained_icnn_weights_filepath is not None:
                    #     print(f'Saving trained weight to ' \
                    #           + f'{self.pretrained_icnn_weights_filepath}')
                    #     geometry.save_weights(self.pretrained_icnn_weights_filepath)
                    # else:
                    #     print(f'Saving trained weight to icnn_weight_trained.pth')
                    #     geometry.save_weights(f'icnn_weight_trained.pth')

        return scalars, meshes

    def forward(self, q: Tensor, v: Tensor,
                u: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Evaluates multibody system dynamics terms at given state and input.

        Calculation is performed as a thin wrapper around
        :py:class:`LagrangianTerms` and :py:class:`ContactTerms`. For
        convenience, this function also returns the Delassus operator
        `D(q) = J(q)^T inv(M(q)) J(q)`.

        Args:
            q: (\*, n_q) configuration batch.
            v: (\*, n_v) velocity batch.
            u: (\*, n_u) input batch.

        Returns:
            (\*, 3 * n_collisions, 3 * n_collisions) Delassus operator D(q).
            (\*, n_v, n_v) mass matrix batch M(q).
            (\*, 3 * n_collisions, n_v) contact Jacobian J(q).
            (\*, n_collisions) signed distance phi(q).
            (\*, n_v) Contact-free acceleration inv(M(q)) * F(q).
        """
        M, non_contact_acceleration = self.lagrangian_terms(q, v, u)
        phi, J, p_BiBc_B, obj_pair_list, R_FW_list, mu_list = \
            self.contact_terms(q)

        delassus = pbmm(J, torch.linalg.solve(M, J.transpose(-1, -2)))
        return delassus, M, J, phi, non_contact_acceleration, p_BiBc_B, \
            obj_pair_list, R_FW_list, mu_list

    def __init__(self, urdfs: Dict[str, str],
                 learnable_body_dict: Dict[str, LearnableBodySettings] = {},
                 pretrained_icnn_weights_filepath: str = None,
                 represent_geometry_as: str = 'box',
                 precomputed_functions: Dict[str, Union[List[str],Callable]]={},
                 export_drake_pytorch_dir: str = None,
                 vis_hook: HookGradientVisualizer = None,
                 ) -> None:
        """Inits :py:class:`MultibodyTerms` for system described in URDFs

        Interpretation is performed as a thin wrapper around
        :py:class:`LagrangianTerms` and :py:class:`ContactTerms`.

        As this module is also responsible for evaluating updated URDF
        representations, the associations between bodies and geometries is
        also tracked to enable URDF rendering in
        ``MultibodyTerms.EvalUrdfRepresentation`` and Tensorboard logging in
        ``MultibodyTerms.scalars``.

        Args:
            urdfs: Dictionary of named URDF XML file names, containing
                description of multibody system.
            learnable_body_dict: specify which body parameters to learn
            pretrained_icnn_weights_filepath: Filepath to a set of
                pretrained ICNN weights.
            represent_geometry_as: String box/mesh/polygon to determine how
                the geometry should be represented.
            precomputed_functions: Dictionary of precomputed functions.  Keys
                that will be considered are 'mass_matrix' and
                'lagrangian_forces'.  The values at those keys are nested
                dictionaries with keys 'function' with the callable and
                'state_names' with a list of strings for the plant's state names
                that were used when the function was pre-computed.  The state
                names are checked to match the state names of the newly created
                plant.
            export_drake_pytorch_dir: The folder in which exported elements of
                the mass matrix and lagrangian force expressions will be saved.
                If provided, the code terminates after the export.
        """
        super().__init__()

        plant_diagram = MultibodyPlantDiagram(urdfs)
        plant = plant_diagram.plant.ToSymbolic()
        inspector = plant_diagram.scene_graph.model_inspector()

        _, all_body_ids = drake_utils.get_all_bodies(plant,
                                                     plant_diagram.model_ids)

        # sweep over collision elements
        geometry_body_assignment: Dict[str, List[int]] = {
            body_id: [] for body_id in all_body_ids
        }

        geometry_ids = plant_diagram.collision_geometry_set.ids

        for geometry_index, geometry_id in enumerate(geometry_ids):
            geometry_frame_id = inspector.GetFrameId(geometry_id)
            geometry_body = plant.GetBodyFromFrameId(geometry_frame_id)
            geometry_body_identifier = drake_utils.unique_body_identifier(
                plant, geometry_body)
            geometry_body_assignment[geometry_body_identifier].append(
                geometry_index)

        self.vis_hook = vis_hook

        # setup parameterization
        self.lagrangian_terms = LagrangianTerms(
            plant_diagram, learnable_body_dict,
            precomputed_functions=precomputed_functions,
            export_drake_pytorch_dir=export_drake_pytorch_dir)
        self.contact_terms = ContactTerms(
            plant_diagram, represent_geometry_as, learnable_body_dict,
            vis_hook=self.vis_hook)
        self.geometry_body_assignment = geometry_body_assignment
        self.plant_diagram = plant_diagram
        self.urdfs = urdfs
        self.pretrained_icnn_weights_filepath = pretrained_icnn_weights_filepath
        self.learnable_body_dict = learnable_body_dict
