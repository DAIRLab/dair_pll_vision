"""Construction and analysis of learnable multibody systems.

Similar to Drake, multibody systems are instantiated as a child class of
:py:class:`System`: :py:class:`MultibodyLearnableSystem`. This object is a thin
wrapper for a :py:class:`MultibodyTerms` member variable, which manages
computation of lumped terms necessary for simulation and evaluation.

Simulation is implemented via Anitescu's [1] convex method.

An interface for the ContactNets [2] loss is also defined as an alternative
to prediction loss.

A large portion of the internal implementation of :py:class:`DrakeSystem` is
implemented in :py:class:`MultibodyPlantDiagram`.

[1] M. Anitescu, “Optimization-based simulation of nonsmooth rigid
multibody dynamics,” Mathematical Programming, 2006,
https://doi.org/10.1007/s10107-005-0590-7

[2] S. Pfrommer*, M. Halm*, and M. Posa. "ContactNets: Learning Discontinuous
Contact Dynamics with Smooth, Implicit Representations," Conference on
Robotic Learning, 2020, https://proceedings.mlr.press/v155/pfrommer21a.html
"""
from multiprocessing import pool
import os
from os import path
import pdb
from typing import List, Tuple, Optional, Dict, cast

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, ParameterList, Parameter
import torch.nn as nn

from dair_pll import urdf_utils, tensor_utils, file_utils
from dair_pll.drake_system import DrakeSystem
from dair_pll.integrator import VelocityIntegrator
from dair_pll.multibody_terms import MultibodyTerms, InertiaLearn
from dair_pll.solvers import DynamicCvxpyLCQPLayer
from dair_pll.state_space import FloatingBaseSpace, StateSpace
from dair_pll.system import System, SystemSummary
from dair_pll.tensor_utils import pbmm, broadcast_lorentz, \
    one_vector_block_diagonal, project_lorentz, reflect_lorentz

from dair_pll import quaternion

# Scaling factors to equalize translation and rotation errors.
# For rotation versus linear scaling:  penalize 0.1 meters same as 90 degrees.
ROTATION_SCALING = 0.2/torch.pi
# For articulation versus linear/rotation scaling:  penalize the scenario where
# one elbow link is in the right place and the other is 180 degrees flipped the
# same, whether link 1 or link 2 are in the right place.
ELBOW_COM_TO_AXIS_DISTANCE = 0.035
JOINT_SCALING = 2*ELBOW_COM_TO_AXIS_DISTANCE/torch.pi + ROTATION_SCALING


class MultibodyLearnableSystem(System):
    """:py:class:`System` interface for dynamics associated with
    :py:class:`MultibodyTerms`."""
    multibody_terms: MultibodyTerms
    init_urdfs: Dict[str, str]
    output_urdfs_dir: Optional[str] = None
    visualization_system: Optional[DrakeSystem]
    solver: DynamicCvxpyLCQPLayer
    dt: float

    def __init__(self,
                 init_urdfs: Dict[str, str],
                 dt: float,
                 loss_weights_dict: dict,
                 inertia_mode: InertiaLearn = InertiaLearn(),
                 constant_bodies: List[str] = [],
                 output_urdfs_dir: Optional[str] = None,
                 pretrained_icnn_weights_filepath: Optional[str] = None,
                 represent_geometry_as: str = 'box') -> None:
        """Inits :py:class:`MultibodyLearnableSystem` with provided model URDFs.

        Implementation is primarily based on Drake. Bodies are modeled via
        :py:class:`MultibodyTerms`, which uses Drake symbolics to generate
        dynamics terms, and the system can be exported back to a
        Drake-interpretable representation as a set of URDFs.

        Args:
            init_urdfs: Names and corresponding URDFs to model with
                :py:class:`MultibodyTerms`.
            dt: Time step of system in seconds.
            inertia_mode: An InertiaLearn() object specifying which inertial
              parameters to learn
            constant_bodies: list of body names whose properties should NOT
              be learned
            loss_weights_dict: Dictionary of weights for the vision loss.
                Requires keys 'w_pred', 'w_comp', 'w_pen', 'w_diss', and
                'w_bsdf'.
            output_urdfs_dir: Optionally, a directory that learned URDFs can be
                written to.
            pretrained_icnn_weights_filepath: Filepath of a set of pretrained
                ICNN weights.
        """

        multibody_terms = MultibodyTerms(
            init_urdfs,
            inertia_mode=inertia_mode,
            constant_bodies=constant_bodies,
            represent_geometry_as=represent_geometry_as,
            pretrained_icnn_weights_filepath=pretrained_icnn_weights_filepath)

        space = multibody_terms.plant_diagram.space
        integrator = VelocityIntegrator(space, self.sim_step, dt)
        super().__init__(space, integrator)
        
        self.output_urdfs_dir = output_urdfs_dir
        self.multibody_terms = multibody_terms
        self.init_urdfs = init_urdfs

        self.visualization_system = None
        self.solver = DynamicCvxpyLCQPLayer(self.space.n_v)
        self.dt = dt
        self.set_carry_sampler(lambda: Tensor([False]))
        self.max_batch_dim = 1

        # Save the loss weights.
        self.w_pred = loss_weights_dict['w_pred']
        self.w_comp = loss_weights_dict['w_comp']
        self.w_pen = loss_weights_dict['w_pen']
        self.w_diss = loss_weights_dict['w_diss']
        self.w_bsdf = loss_weights_dict['w_bsdf']

    def generate_updated_urdfs(self, suffix: str = None) -> Dict[str, str]:
        """Exports current parameterization as a :py:class:`DrakeSystem`.

        Args:
            storage_name: name of file storage location in which to store new
              URDFs for Drake to read.
            suffix: optionally can include a suffix for generated filename.

        Returns:
            New Drake system instantiated on new URDFs.
        """
        assert self.output_urdfs_dir is not None
        old_urdfs = self.init_urdfs
        new_urdf_strings = urdf_utils.represent_multibody_terms_as_urdfs(
            self.multibody_terms, self.output_urdfs_dir)
        new_urdfs = {}

        # Save new urdfs with original file basenames plus optional suffix in
        # new folder.
        for urdf_name, new_urdf_string in new_urdf_strings.items():
            old_urdf_filename = path.basename(old_urdfs[urdf_name])

            if suffix is not None:
                # Rename test.obj to test_{suffix}.obj.
                obj_file = os.path.join(self.output_urdfs_dir, 'test.obj')
                new_obj_file = os.path.join(
                    self.output_urdfs_dir, f'test_{suffix}.obj')
                os.rename(obj_file, new_obj_file)
                
                # Replace references in the urdf to the new filename.
                new_urdf_string = new_urdf_string.replace(
                    'test.obj', f'test_{suffix}.obj')
                
            new_urdf_path = path.join(self.output_urdfs_dir, old_urdf_filename)
            file_utils.save_string(new_urdf_path, new_urdf_string)
            new_urdfs[urdf_name] = new_urdf_path

        return new_urdfs

    def contactnets_loss(self,
                         x: Tensor,
                         u: Tensor,
                         x_plus: Tensor,
                         loss_pool: Optional[pool.Pool] = None) -> Tensor:
        r"""Calculate ContactNets [1] loss for state transition.

        Change made to scale this loss to be per kilogram.  This helps prevent
        sending mass quantities to zero in multibody learning scenarios.

        References:
            [1] S. Pfrommer*, M. Halm*, and M. Posa. "ContactNets: Learning
            Discontinuous Contact Dynamics with Smooth, Implicit
            Representations," Conference on Robotic Learning, 2020,
            https://proceedings.mlr.press/v155/pfrommer21a.html

        Args:
            x: (\*, space.n_x) current state batch.
            u: (\*, ?) input batch.
            x_plus: (\*, space.n_x) current state batch.
            loss_pool: optional processing pool to enable multithreaded solves.

        Returns:
            (\*,) loss batch.
        """
        loss_pred, loss_comp, loss_pen, loss_diss = \
            self.calculate_contactnets_loss_terms(x, u, x_plus)

        loss = (self.w_pred * loss_pred) + (self.w_comp * loss_comp) + \
               (self.w_pen * loss_pen) + (self.w_diss * loss_diss)

        return loss

    def calculate_contactnets_loss_terms(
            self, x: Tensor, u: Tensor, x_plus: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Helper function for
        :py:meth:`MultibodyLearnableSystem.contactnets_loss` that returns the
        individual pre-weighted loss contributions:
            * Prediction
            * Complementarity
            * Penetration
            * Dissipation

        Args:
            x: (*, space.n_x) current state batch.
            u: (*, ?) input batch.
            x_plus: (*, space.n_x) current state batch.

        Returns:
            (*,) prediction error loss.
            (*,) complementarity violation loss.
            (*,) penetration loss.
            (*,) dissipation violation loss.
        """
        if self.w_pred == 0:
            assert self.w_comp==self.w_pen==self.w_diss==0, f'w_pred is 0 ' + \
                f'so required the rest are zero too.'
            return torch.zeros_like(x[..., 0]), torch.zeros_like(x[..., 0]), \
                   torch.zeros_like(x[..., 0]), torch.zeros_like(x[..., 0])

        # pylint: disable-msg=too-many-locals
        v = self.space.v(x)
        q_plus, v_plus = self.space.q_v(x_plus)
        dt = self.dt

        # Begin loss calculation.
        delassus, M, J, phi, non_contact_acceleration, _p_BiBc_B, \
            _obj_pair_list, _R_FW_list = \
                self.multibody_terms(q_plus, v_plus, u)

        # Construct a reordering matrix s.t. lambda_CN = reorder_mat @ f_sappy.
        n_contacts = phi.shape[-1]
        reorder_mat = tensor_utils.sappy_reorder_mat(n_contacts)
        reorder_mat = reorder_mat.reshape((1,) * (delassus.dim() - 2) +
                                          reorder_mat.shape).expand(
                                              delassus.shape)
        J_t = J[..., n_contacts:, :]

        # pylint: disable=E1103
        double_zero_vector = torch.zeros(phi.shape[:-1] + (2 * n_contacts,))
        phi_then_zero = torch.cat((phi, double_zero_vector), dim=-1)

        # pylint: disable=E1103
        sliding_velocities = pbmm(J_t, v_plus.unsqueeze(-1))
        sliding_speeds = sliding_velocities.reshape(
            phi.shape[:-1] + (n_contacts, 2)).norm(dim=-1, keepdim=True)

        Q = delassus
        J_M = pbmm(reorder_mat.transpose(-1, -2),
                   pbmm(J, torch.linalg.cholesky(torch.inverse((M)))))

        dv = (v_plus - (v + non_contact_acceleration * dt)).unsqueeze(-2)

        q_pred = -pbmm(J, dv.transpose(-1, -2))
        q_comp = torch.abs(phi_then_zero).unsqueeze(-1)
        q_diss = dt * torch.cat((sliding_speeds, sliding_velocities), dim=-2)

        # Implement ContactNets loss weighting -- for solving for the forces,
        # normalize based on w_pred.
        q = q_pred + (self.w_comp/self.w_pred)*q_comp + \
                     (self.w_diss/self.w_pred)*q_diss

        c_pen = (torch.maximum(-phi, torch.zeros_like(phi))**2).sum(dim=-1)
        c_pen = c_pen.reshape(c_pen.shape + (1, 1))

        c_pred = 0.5 * pbmm(dv, pbmm(M, dv.transpose(-1, -2)))

        # Envelope theorem guarantees that gradient of loss w.r.t. parameters
        # can ignore the gradient of the impulses w.r.t. the QCQP parameters.
        # Therefore, we can detach ``impulses`` from pytorch's computation graph
        # without causing error in the overall loss gradient.
        # pylint: disable=E1103
        try:
            impulses = pbmm(
                reorder_mat,
                self.solver(
                    J_M,
                    pbmm(reorder_mat.transpose(-1, -2), q).squeeze(-1)
                ).detach().unsqueeze(-1))
        except:
            pdb.set_trace()

        # Hack: remove elements of ``impulses`` where solver likely failed.
        invalid = torch.any(
            (impulses.abs() > 1e3) | impulses.isnan() | impulses.isinf(),
            dim=-2, keepdim=True)

        c_pred[invalid] *= 0.
        c_pen[invalid] *= 0.
        impulses[invalid.expand(impulses.shape)] = 0.

        loss_pred = 0.5 * pbmm(impulses.transpose(-1, -2), pbmm(Q, impulses)) \
                    + pbmm(impulses.transpose(-1, -2), q_pred) + c_pred
        loss_comp = pbmm(impulses.transpose(-1, -2), q_comp)
        loss_pen = c_pen
        loss_diss = pbmm(impulses.transpose(-1, -2), q_diss)

        return loss_pred.reshape(-1), loss_comp.reshape(-1), \
               loss_pen.reshape(-1), loss_diss.reshape(-1)

    def forward_dynamics(self,
                         q: Tensor,
                         v: Tensor,
                         u: Tensor,
                         dynamics_pool: Optional[pool.Pool] = None) -> Tensor:
        r"""Calculates delta velocity from current state and input.

        Implements Anitescu's [1] convex formulation in dual form, derived
        similarly to Tedrake [2] and described here.

        Let v_minus be the contact-free next velocity, i.e.::

            v + dt * non_contact_acceleration.

        Let FC be the combined friction cone::

            FC = {[beta_n beta_t]: beta_n_i >= ||beta_t_i||}.

        The primal version of Anitescu's formulation is as follows::

            min_{v_plus,s}  (v_plus - v_minus)^T M(q)(v_plus - v_minus)/2
            s.t.            s = [I; 0]phi(q)/dt + J(q)v_plus,
                            s \\in FC.

        The KKT conditions are the mixed cone complementarity
        problem [3, Theorem 2]::

            s = [I; 0]phi(q)/dt + J(q)v_plus,
            M(q)(v_plus - v_minus) = J(q)^T f,
            FC \\ni s \\perp f \\in FC.

        As M(q) is positive definite, we can solve for v_plus in terms of
        lambda, and thus these conditions can be simplified to::

            FC \\ni D(q)f + J(q)v_minus + [I;0]phi(q)/dt \\perp f \\in FC.

        which in turn are the KKT conditions for the dual QCQP we solve::

            min_{f}     f^T D(q) f/2 + f^T(J(q)v_minus + [I;0]phi(q)/dt)
            s.t.        f \\in FC.

        References:
            [1] M. Anitescu, “Optimization-based simulation of nonsmooth rigid
            multibody dynamics,” Mathematical Programming, 2006,
            https://doi.org/10.1007/s10107-005-0590-7

            [2] R. Tedrake. Underactuated Robotics: Algorithms for Walking,
            Running, Swimming, Flying, and Manipulation (Course Notes for MIT
            6.832), https://underactuated.mit.edu

            [3] S. Z. N'emeth, G. Zhang, "Conic optimization and
            complementarity problems," arXiv,
            https://doi.org/10.48550/arXiv.1607.05161
        Args:
            q: (\*, space.n_q) current configuration batch.
            v: (\*, space.n_v) current velocity batch.
            u: (\*, ?) current input batch.
            dynamics_pool: optional processing pool to enable multithreaded
              solves.

        Returns:
            (\*, space.n_v) delta velocity batch.
        """
        # pylint: disable=too-many-locals
        dt = self.dt
        phi_eps = 1e6
        delassus, M, J, phi, non_contact_acceleration, _p_BiBc_B, \
            _obj_pair_list, _R_FW_list = \
                self.multibody_terms(q, v, u)
        n_contacts = phi.shape[-1]
        contact_filter = (broadcast_lorentz(phi) <= phi_eps).unsqueeze(-1)

        reorder_mat = tensor_utils.sappy_reorder_mat(n_contacts)
        reorder_mat = reorder_mat.reshape((1,) * (delassus.dim() - 2) +
                                          reorder_mat.shape).expand(
                                              delassus.shape)
        J_M = pbmm(reorder_mat.transpose(-1, -2),
                   pbmm(J, torch.linalg.cholesky(torch.inverse((M)))))

        # pylint: disable=E1103
        double_zero_vector = torch.zeros(phi.shape[:-1] + (2 * n_contacts,))
        phi_then_zero = torch.cat((phi, double_zero_vector),
                                  dim=-1).unsqueeze(-1)

        v_minus = v + dt * non_contact_acceleration
        q_full = pbmm(J, v_minus.unsqueeze(-1)) + (1 / dt) * phi_then_zero

        impulse_full = pbmm(
            reorder_mat,
            self.solver(
                J_M,
                pbmm(reorder_mat.transpose(-1, -2), q_full).squeeze(-1)
            ).unsqueeze(-1))

        impulse = torch.zeros_like(impulse_full)
        impulse[contact_filter] += impulse_full[contact_filter]

        return v_minus + torch.linalg.solve(M, pbmm(J.transpose(-1, -2),
                                                    impulse)).squeeze(-1)

    def sim_step(self, x: Tensor, carry: Tensor) -> Tuple[Tensor, Tensor]:
        """``Integrator.partial_step`` wrapper for
        :py:meth:`forward_dynamics`."""
        q, v = self.space.q_v(x)
        # pylint: disable=E1103
        u = torch.zeros(q.shape[:-1] + (0,))
        v_plus = self.forward_dynamics(q, v, u)
        return v_plus, carry

    def summary(self, statistics: Dict) -> SystemSummary:
        """Generates summary statistics for multibody system.

        The scalars returned are simply the scalar description of the
        system's :py:class:`MultibodyTerms`.

        Meshes are generated for learned
        :py:class:`~dair_pll.geometry.DeepSupportConvex` es.

        Args:
            statistics: Updated evaluation statistics for the model.

        Returns:
            Scalars and meshes packaged into a ``SystemSummary``.
        """
        scalars, meshes = self.multibody_terms.scalars_and_meshes()
        videos = cast(Dict[str, Tuple[np.ndarray, int]], {})

        return SystemSummary(scalars=scalars, videos=videos, meshes=meshes)

    def bundlesdf_data_generation_from_cnets(self,
                         x: Tensor,
                         u: Tensor,
                         x_plus: Tensor,
                         loss_pool: Optional[pool.Pool] = None):
        # pylint: disable-msg=too-many-locals
        v = self.space.v(x)
        q_plus, v_plus = self.space.q_v(x_plus)
        dt = self.dt

        delassus, M, J, phi, non_contact_acceleration, p_BiBc_B, \
            _obj_pair_list, _R_FW_list = self.multibody_terms(q_plus, v_plus, u)

        n_contacts = phi.shape[-1]
        reorder_mat = tensor_utils.sappy_reorder_mat(n_contacts)
        reorder_mat = reorder_mat.reshape((1,) * (delassus.dim() - 2) +
                                          reorder_mat.shape).expand(
                                              delassus.shape)
        J_t = J[..., n_contacts:, :]

        # pylint: disable=E1103
        double_zero_vector = torch.zeros(phi.shape[:-1] + (2 * n_contacts,))
        phi_then_zero = torch.cat((phi, double_zero_vector), dim=-1)

        # pylint: disable=E1103
        sliding_velocities = pbmm(J_t, v_plus.unsqueeze(-1))
        sliding_speeds = sliding_velocities.reshape(phi.shape[:-1] +
                                                    (n_contacts, 2)).norm(
                                                        dim=-1, keepdim=True)

        J_M = pbmm(reorder_mat.transpose(-1, -2),
                   pbmm(J, torch.linalg.cholesky(torch.inverse((M)))))

        dv = (v_plus - (v + non_contact_acceleration * dt)).unsqueeze(-2)

        q_pred = -pbmm(J, dv.transpose(-1, -2))
        q_comp = torch.abs(phi_then_zero).unsqueeze(-1)
        q_diss = dt * torch.cat((sliding_speeds, sliding_velocities), dim=-2)
        q = q_pred + q_comp + q_diss

        penetration_penalty = (torch.maximum(
            -phi, torch.zeros_like(phi))**2).sum(dim=-1)

        penetration_penalty = penetration_penalty.reshape(
            penetration_penalty.shape + (1, 1)) * 100.

        constant = 0.5 * pbmm(dv, pbmm(M, dv.transpose(
            -1, -2))) + penetration_penalty

        # Envelope theorem guarantees that gradient of loss w.r.t. parameters
        # can ignore the gradient of the force w.r.t. the QCQP parameters.
        # Therefore, we can detach ``force`` from pytorch's computation graph
        # without causing error in the overall loss gradient.
        # pylint: disable=E1103
        impulses = pbmm(
            reorder_mat,
            self.solver(
                J_M,
                pbmm(reorder_mat.transpose(-1, -2), q).squeeze(-1),
            ).detach().unsqueeze(-1))

        # Hack: remove elements of ``impulses`` where solver likely failed.
        invalid = torch.any(
            (impulses.abs() > 1e3) | impulses.isnan() | impulses.isinf(),
            dim=-2, keepdim=True)

        constant[invalid] *= 0.
        impulses[invalid.expand(impulses.shape)] = 0.

        # Get the normal forces
        normal_impulses = impulses[:, :n_contacts].reshape(-1, n_contacts)
        orientation = q_plus[..., :4]

        # Get the contact points that correspond to high normal forces
        def ground_orientation_in_body_frame(object_orientation, n_lambda):
            """
            Convert ground orientation from world frame to object's body frame.
            """
            n_hat = torch.tensor([.0,.0,-1.0])
            n_hat_repeated = torch.tile(n_hat.unsqueeze(0), (n_lambda,1))
            return quaternion.rotate(quaternion.inverse(object_orientation),
                                     n_hat_repeated)
        
        points, directions = torch.zeros((0,3)), torch.zeros((0,3))
        impulses_flat = torch.zeros((0))
        states = torch.zeros((0, self.space.n_x))
        n_lambda = normal_impulses.shape[1]
        
        orientation = torch.tile(orientation.unsqueeze(1), (1, n_lambda, 1))
        state = torch.tile(x_plus.unsqueeze(1), (1, n_lambda, 1))
        for force_i, points_i, orientation_i, state_i in \
            zip(normal_impulses, p_BiBc_B, orientation, state):
            
            support_points = points_i
            orientation_i = ground_orientation_in_body_frame(orientation_i,
                                                             n_lambda)
            support_function = orientation_i
            points = torch.cat((points, support_points), dim=0)
            directions = torch.cat((directions, support_function), dim=0)
            impulses_flat = torch.cat((impulses_flat, force_i), dim=0)
            states = torch.cat((states, state_i), dim=0)

        return points, directions, impulses_flat/self.dt, states
    