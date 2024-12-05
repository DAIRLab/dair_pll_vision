"""Collision geometry representation for multibody systems.

Each type of collision geometry is modeled as a class inheriting from the
``CollisionGeometry`` abstract type. Different types of inheriting geometries
will need to resolve collisions in unique ways, but one interface is always
expected: a list of scalars to track during training.

Many collision geometries, such as boxes and cylinders, fall into the class
of bounded (compact) convex shapes. A general interface is defined in the
abstract ``BoundedConvexCollisionGeometry`` type, which returns a set of
witness points given support hyperplane directions. One implementation is
the ``SparseVertexConvexCollisionGeometry`` type, which finds these points
via brute force optimization over a short list of vertices.

All collision geometries implemented here mirror a Drake ``Shape`` object. A
general purpose converter is implemented in
``PydrakeToCollisionGeometryFactory``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Dict, cast, Union, Optional
import pdb

import fcl  # type: ignore
import numpy as np
import pywavefront  # type: ignore
import torch
import trimesh
from pydrake.geometry import Box as DrakeBox  # type: ignore
from pydrake.geometry import Sphere as DrakeSphere # type: ignore
from pydrake.geometry import HalfSpace as DrakeHalfSpace  # type: ignore
from pydrake.geometry import Mesh as DrakeMesh  # type: ignore
from pydrake.geometry import Shape  # type: ignore
from torch import Tensor
from torch.nn import Module, Parameter
from scipy.spatial import ConvexHull

from dair_pll.deep_support_function import HomogeneousICNN, \
    extract_mesh_from_support_function
from dair_pll.tensor_utils import pbmm, tile_dim, \
    rotation_matrix_from_one_vector

_UNIT_BOX_VERTICES = Tensor([[0, 0, 0, 0, 1, 1, 1, 1.], [
    0, 0, 1, 1, 0, 0, 1, 1.
], [0, 1, 0, 1, 0, 1, 0, 1.]]).t() * 2. - 1.

_NOMINAL_HALF_LENGTH = 0.05   # 10cm is nominal object length

_total_ordering = ['Plane', 'Box', 'Sphere', 'Polygon', 'DeepSupportConvex']

_POLYGON_DEFAULT_N_QUERY = 5
_DEEP_SUPPORT_DEFAULT_N_QUERY = 5
_DEEP_SUPPORT_EVAL_N_QUERY = 10
_DEEP_SUPPORT_DEFAULT_DEPTH = 2
_DEEP_SUPPORT_DEFAULT_WIDTH = 256

DEBUG_TRIMESH_COLLISIONS = False



class CollisionGeometry(ABC, Module):
    """Abstract interface for collision geometries.

    Collision geometries have heterogeneous implementation depending on the
    type of shape. This class mainly enforces the implementation of
    bookkeeping interfaces.

    When two collision geometries are evaluated for collision with
    ``GeometryCollider``, their ordering is constrained via a total order on
    collision geometry types, enforced with an overload of the ``>`` operator.
    """

    name: str = ""
    learnable: bool = False

    def __ge__(self, other) -> bool:
        """Evaluate total ordering of two geometries based on their types."""
        return _total_ordering.index(
            type(self).__name__) > _total_ordering.index(type(other).__name__)

    def __lt__(self, other) -> bool:
        """Evaluate total ordering of two geometries via passthrough to
        ``CollisionGeometry.__ge__()``."""
        return other.__ge__(self)

    @abstractmethod
    def scalars(self) -> Dict[str, float]:
        """Describes object via Tensorboard scalars.

        Any namespace for the object (e.g. "object_5") is assumed to be added by
        external code before adding to Tensorboard, so the names of the
        scalars can be natural descriptions of the geometry's parameters.

        Examples:
            A cylinder might be represented with the following output::

                {'radius': 5.2, 'height': 4.1}

        Returns:
            A dictionary of named parameters describing the geometry.
        """


class Plane(CollisionGeometry):
    """Half-space/plane collision geometry.

    ``Plane`` geometries are assumed to be the plane z=0 in local (i.e.
    "body-axes" coordinates). Any tilted/raised/lowered half spaces are expected
    to be derived by placing the ``z=0`` plane in a rigidly-transformed
    frame."""

    def scalars(self) -> Dict[str, float]:
        """As the plane is fixed to be z=0, there are no parameters."""
        return {}


class BoundedConvexCollisionGeometry(CollisionGeometry):
    """Interface for compact-convex collision geometries.

    Such shapes have a representation via a "support function" h(d),
    which takes in a hyperplane direction and returns how far the shape S
    extends in that direction, i.e.::

        h(d) = max_{s \\in S} s \\cdot d.

    This representation can be leveraged to derive "witness points" -- i.e.
    the closest point(s) between the ``BoundedConvexCollisionGeometry`` and
    another convex shape, such as another ``BoundedConvexCollisionGeometry``
    or a ``Plane``.
    """

    @abstractmethod
    def support_points(
        self, directions: Tensor, hint: Optional[Tensor] = None) -> Tensor:
        """Returns a set of witness points representing contact with another
        shape off in the direction(s) ``directions``.

        This method will return a set of points ``S' \\subset S`` such that::

            argmax_{s \\in S} s \\cdot directions \\subset convexHull(S').


        In theory, returning exactly the argmax set would be sufficient.
        However,

        Args:
            directions: (\*, 3) batch of unit-length directions.
            hint: (\*, 3) batch of expected contact point

        Returns:
            (\*, N, 3) sets of corresponding witness points of cardinality N.
            If hint is defined, then N == 1.
        """

    @abstractmethod
    def get_fcl_geometry(self) -> fcl.CollisionGeometry:
        """Retrieves :py:mod:`fcl` collision geometry representation.

        Returns:
            :py:mod:`fcl` bounding volume
        """


class SparseVertexConvexCollisionGeometry(BoundedConvexCollisionGeometry):
    """Partial implementation of ``BoundedConvexCollisionGeometry`` when
    witness points are guaranteed to be contained in a small set of vertices.

    An obvious subtype is any sort of polytope, such as a ``Box``. A less
    obvious subtype are shapes in which a direction-dependent set of vertices
    can be easily calculated. See ``Cylinder``, for instance.
    """

    def __init__(self, n_query: int, learnable: bool = True) -> None:
        """Inits ``SparseVertexConvexCollisionGeometry`` with prescribed
        query interface.

        Args:
            n_query: number of vertices to return in witness point set.
        """
        super().__init__()
        self.n_query = n_query
        self.learnable = learnable

    def support_points(self, directions: Tensor,
                       hint: Optional[Tensor] = None) -> Tensor:
        """Implements ``BoundedConvexCollisionGeometry.support_points()`` via
        brute force optimization over the witness vertex set.

        Specifically, if S_v is the vertex set, this function returns
        ``n_query`` elements s of S_v for which ``s \\cdot directions`` is
        highest. This set is not guaranteed to be sorted.

        Given the prescribed behavior of
        ``BoundedConvexCollisionGeometry.support_points()``, an implicit
        assumption of this implementation is that the convex hull of the top
        ``n_query`` points in S_v contains ``argmax_{s \\in S} s \\cdot
        directions``.

        Args:
            directions: (\*, 3) batch of directions
            hint: (\*, 3) expected contact point, should be on convex set
            of the witness points. Used if n_query > 1

        Returns:
            (\*, n_query, 3) sets of corresponding witness points.
        """
        assert directions.shape[-1] == 3
        original_shape = directions.shape

        # reshape to (product(*),3)
        directions = directions.view(-1, 3)

        # pylint: disable=E1103
        batch_range = torch.arange(directions.shape[0])
        vertices = self.get_vertices(directions)
        dots = pbmm(directions.unsqueeze(-2),
                    vertices.transpose(-1, -2)).squeeze(-2)

        # top dot product indices in shape (product(*), n_query)
        # pylint: disable=E1103
        selections = torch.topk(dots, self.n_query, dim=-1,
                                sorted=False).indices.t()

        top_vertices = torch.stack(
            [vertices[batch_range, selection] for selection in selections], -2)
        # reshape to (*, n_query, 3)
        queries = top_vertices.view(original_shape[:-1] + (self.n_query, 3))
        if self.n_query > 1 and (hint is not None) and \
            (hint.shape == directions.shape):
            # Find linear combination of queries
            # Lst Sq: queries (*, 3, n_query) * ? (*, n_query, 1) == hint
            # (*, 1, 3)
            # NOTE:  Solution is detached from the gradient chain.
            sol = torch.linalg.lstsq(
                queries.detach().transpose(-1, -2), hint.unsqueeze(-1)).solution
            return pbmm(queries.transpose(-1, -2), sol).transpose(-1, -2)

        return queries

    @abstractmethod
    def get_vertices(self, directions: Tensor, *kwargs: None) -> Tensor:
        """Returns sparse witness point set as collection of vertices.

        Specifically, given search directions, returns a set of points
        ``S_v`` for which::

            argmax_{s \\in S} s \\cdot directions \\subset convexHull(S_v).

        Args:
            directions: (\*, 3) batch of unit-length directions.
        Returns:
            (\*, N, 3) witness point convex hull vertices.
        """


class Polygon(SparseVertexConvexCollisionGeometry):
    """Concrete implementation of a convex polytope.

    Implemented via ``SparseVertexConvexCollisionGeometry`` as a static set
    of vertices, where models the underlying shape as all convex combinations
    of the vertices.
    """
    vertices_parameter: Parameter

    def __init__(self,
                 vertices: Tensor,
                 n_query: int = _POLYGON_DEFAULT_N_QUERY,
                 learnable: bool = True) -> None:
        """Inits ``Polygon`` object with initial vertex set.  Learns vertex set
        in normalized space via _NOMINAL_HALF_LENGTH.  The `get_vertices` method
        returns the vertices in real meter units.

        Args:
            vertices: (N, 3) static vertex set.
            n_query: number of vertices to return in witness point set.
        """
        super().__init__(n_query)

        mesh = trimesh.Trimesh(vertices.numpy(), [])
        hull = mesh.convex_hull
        hull_vertices = Tensor(hull.vertices)/_NOMINAL_HALF_LENGTH
        self.vertices_parameter = Parameter(
            hull_vertices.clone(), requires_grad=learnable)

    def get_fcl_geometry(self) -> fcl.CollisionGeometry:
        raise NotImplementedError

    def get_vertices(self, directions: Tensor, *kwargs: None) -> Tensor:
        """Return batched view of static vertex set"""
        scaled_vertices = _NOMINAL_HALF_LENGTH * self.vertices_parameter
        return scaled_vertices.expand(
            directions.shape[:-1] + scaled_vertices.shape)

    def scalars(self) -> Dict[str, float]:
        """Return one scalar for each vertex index."""
        scalars = {}
        axes = ['x', 'y', 'z']

        # Use arbitrary direction to query the Polygon's vertices (value does
        # not matter).
        arbitrary_direction = torch.ones((1,3))
        vertices = self.get_vertices(arbitrary_direction).squeeze(0)

        for axis, values in zip(axes, vertices.t()):
            for vertex_index, value in enumerate(values):
                scalars[f'v{vertex_index}_{axis}'] = value.item()
        return scalars


class DeepSupportConvex(SparseVertexConvexCollisionGeometry):
    r"""Deep support function convex shape.

    Any convex shape :math:`S` can be equivalently represented via its support
    function :math:`f(d)`, which returns the extent to which the object
    extends in the :math:`d` direction:

    .. math::

        f(d) = \max_{s \in S} s \cdot d.

    Given a direction, the set of points that form the :math:`\arg\max` in
    :math:`f(d)` is exactly the convex subgradient :math:`\partial_d f(d)`.

    Furthermore, for every convex shape, :math:`f(d)` is convex and
    positively homogeneous, and every convex and positively homogeneous
    :math:`f(d)` is the support function of some convex shape.

    This collision geometry type implements the support function directly as
    a convex and positively homogeneous neural network (
    :py:class:`~dair_pll.deep_support_function.HomogeneousICNN`\)."""
    network: HomogeneousICNN
    """Support function representation as a neural net."""
    perturbations: Tensor
    """Perturbed support directions, which aid mesh-plane contact stability."""
    fcl_geometry: fcl.BVHModel
    r""":py:mod:`fcl` mesh collision geometry representation."""

    def __init__(self,
                 vertices: Tensor,
                 n_query: int = _DEEP_SUPPORT_DEFAULT_N_QUERY,
                 depth: int = _DEEP_SUPPORT_DEFAULT_DEPTH,
                 width: int = _DEEP_SUPPORT_DEFAULT_WIDTH,
                 perturbation: float = 0.5,
                 learnable: bool = True) -> None:
        r"""Inits ``DeepSupportConvex`` object with initial vertex set.

        When calculating a sparse vertex set with :py:meth:`get_vertices`,
        supplements the support direction with nearby directions randomly.

        Args:
            vertices: ``(N, 3)`` initial vertex set.
            n_query: Number of vertices to return in witness point set.
            depth: Depth of support function network.
            width: Width of support function network.
            perturbation: support direction sampling parameter.
        """
        # pylint: disable=too-many-arguments,E1103
        super().__init__(n_query)
        length_scale = (vertices.max(dim=0).values -
                        vertices.min(dim=0).values).norm() / 2
        self.network = HomogeneousICNN(
            depth, width, scale=length_scale, learnable=learnable)
        self.perturbations = torch.cat((torch.zeros(
            (1, 3)), perturbation * (torch.rand((n_query - 1, 3)) - 0.5)))

    def get_vertices(
            self, directions: Tensor, sample_entire_mesh: bool = False
    ) -> Tensor:
        """Return batched view of support points of interest.

        Given a direction :math:`d`, this function finds the support point of
        the object in that direction, calculated via envelope theorem.  If
        ``sample_entire_mesh`` is set to True, this function samples points
        broadly across the whole surface of the object geometry.

        Args:
            directions: ``(*, 3)`` batch of support directions sample.

        Returns:
            ``(*, n_query, 3)`` sampled support points.
        """
        if sample_entire_mesh:
            mesh = extract_mesh_from_support_function(self.network)
            single_vertices = mesh.vertices
            return single_vertices.expand(
                directions.shape[:-1] + single_vertices.shape)

        # Can query different number of directions during training/evaluation.
        n_to_add = self.n_query if self.network.training else \
            _DEEP_SUPPORT_EVAL_N_QUERY

        perturbed = directions.unsqueeze(-2)
        perturbed = tile_dim(perturbed, n_to_add, -2)

        perturbed += torch.cat((torch.zeros(
                (1, 3)), 1.99 * (torch.rand((n_to_add - 1, 3)) - 0.5)
            )).expand(perturbed.shape)
        # 1.99*[-0.5, 0.5]=(-1, 1), which is added to unit-length vectors,
        # allowing the perturbations to cover a hemisphere. 
        # This is intentional to allow wider coverage of the support function.
        perturbed /= perturbed.norm(dim=-1, keepdim=True)

        return self.network(perturbed)

    def train(self, mode: bool = True) -> DeepSupportConvex:
        r"""Override training-mode setter from :py:mod:`torch`.

        Sets a static fcl mesh geometry for the entirety of evaluation time,
        as the underlying support function is not changing.

        Args:
            mode: ``True`` for training, ``False`` for evaluation.

        Returns:
            ``self``.
        """
        if not mode:
            self.fcl_geometry = self.get_fcl_geometry()
        return cast(DeepSupportConvex, super().train(mode))

    def get_fcl_geometry(self) -> fcl.CollisionGeometry:
        """Retrieves :py:mod:`fcl` mesh collision geometry representation.

        If evaluation mode is set, retrieves precalculated version.

        Returns:
            :py:mod:`fcl` bounding volume hierarchy for mesh.
        """
        if self.training:
            mesh = extract_mesh_from_support_function(self.network)
            vertices = mesh.vertices.numpy()
            faces = mesh.faces.numpy()
            self.fcl_geometry = fcl.BVHModel()
            self.fcl_geometry.beginModel(vertices.shape[0], faces.shape[0])
            self.fcl_geometry.addSubModel(vertices, faces)
            self.fcl_geometry.endModel()

        return self.fcl_geometry

    def scalars(self) -> Dict[str, float]:
        """no scalars!"""
        return {}

    def save_weights(self, file_path: str) -> None:
        torch.save(self.network.state_dict(), file_path)

    def load_weights(self, file_path: str) -> None:
        self.network.load_state_dict(torch.load(file_path))
        self.network.train()


class Box(SparseVertexConvexCollisionGeometry):
    """Implementation of cuboid geometry as a sparse vertex convex hull.

    To prevent learning negative box lengths, the learned parameters are stored
    as :py:attr:`length_params`, and the box's half lengths can be computed
    as their absolute value.  The desired half lengths can be accessed via
    :py:meth:`get_half_lengths`.
    """
    length_params: Parameter
    unit_vertices: Tensor

    def __init__(self, half_lengths: Tensor, n_query: int,
                 learnable: bool = True) -> None:
        """Inits ``Box`` object with initial size.

        Args:
            half_lengths: (3,) half-length dimensions of box on x, y,
              and z axes.
            n_query: number of vertices to return in witness point set.
        """
        super().__init__(n_query)

        assert half_lengths.numel() == 3

        scaled_half_lengths = half_lengths.clone()/_NOMINAL_HALF_LENGTH
        self.length_params = Parameter(scaled_half_lengths.view(1, -1),
                                       requires_grad=learnable)
        self.unit_vertices = _UNIT_BOX_VERTICES.clone()

    def get_half_lengths(self) -> Tensor:
        """From the stored :py:attr:`length_params`, compute the half lengths of
        the box as its absolute value."""
        return torch.abs(self.length_params) * _NOMINAL_HALF_LENGTH

    def get_vertices(self, directions: Tensor, *kwargs: None) -> Tensor:
        """Returns view of cuboid's static vertex set."""
        return (self.unit_vertices *
                self.get_half_lengths()).expand(directions.shape[:-1] +
                                                self.unit_vertices.shape)

    def scalars(self) -> Dict[str, float]:
        """Returns each axis's full length as a scalar."""
        scalars = {
            f'len_{axis}': 2 * value.item()
            for axis, value in zip(['x', 'y', 'z'],
                self.get_half_lengths().view(-1))
        }
        return scalars

    def get_fcl_geometry(self) -> fcl.CollisionGeometry:
        """Retrieves :py:mod:`fcl` collision geometry representation.

        Returns:
            :py:mod:`fcl` bounding volume
        """
        scalars = self.scalars()

        return fcl.Box(scalars["len_x"], scalars["len_y"], scalars["len_z"])


class Sphere(BoundedConvexCollisionGeometry):
    """Implements sphere geometry via its support function.

    It is trivial to calculate the witness point for a sphere contact as
    simply the product of the sphere's radius and the support direction.

    To prevent learning a negative radius, the learned parameter is stored as
    :py:attr:`length_param`, and the sphere's radius can be computed as its
    absolute value.  The desired radius can be accessed via
    :py:meth:`get_radius`.
    """
    length_param: Parameter

    def __init__(self, radius: Tensor, learnable: bool = True) -> None:
        super().__init__()
        assert radius.numel() == 1

        self.length_param = Parameter(radius.clone().view(()),
                                      requires_grad=learnable)
        self.learnable = learnable

    def get_radius(self) -> Tensor:
        """From the stored :py:attr:`length_param`, compute the radius of the
        sphere as its absolute value."""
        return torch.abs(self.length_param)

    def support_points(
            self, directions: Tensor, _: Optional[Tensor] = None) -> Tensor:
        """Implements ``BoundedConvexCollisionGeometry.support_points()``
        via analytic expression::

            argmax_{s \\in S} s \\cdot directions = directions * radius.

        Args:
            directions: (\*, 3) batch of directions.

        Returns:
            (\*, 1, 3) corresponding witness point sets of cardinality 1.
        """
        return (directions.clone() * self.get_radius()).unsqueeze(-2)

    def scalars(self) -> Dict[str, float]:
        """Logs radius as a scalar."""
        return {'radius': self.get_radius().item()}

    def get_fcl_geometry(self) -> fcl.CollisionGeometry:
        """Retrieves :py:mod:`fcl` collision geometry representation.

        Returns:
            :py:mod:`fcl` bounding volume
        """
        scalars = self.scalars()

        return fcl.Sphere(scalars["radius"])


class PydrakeToCollisionGeometryFactory:
    """Utility class for converting Drake ``Shape`` instances to
    ``CollisionGeometry`` instances."""

    @staticmethod
    def convert(drake_shape: Shape, represent_geometry_as: str,
        learnable: bool = True, name: str = ""
        ) -> CollisionGeometry:
        """Converts abstract ``pydrake.geometry.shape`` to
        ``CollisionGeometry`` according to the desired ``represent_geometry_as``
        type.  If the body is not learnable, then it will use the default
        simplest type, regardless of ``represent_geometry_as``.

        Notes:
            The desired ``represent_geometry_as`` type only will affect
            ``DrakeBox`` and ``DrakeMesh`` types, not ``DrakeHalfSpace`` types.

        Args:
            drake_shape: drake shape type to convert.

        Returns:
            Collision geometry representation of shape.

        Raises:
            TypeError: When provided object is not a supported Drake shape type.
        """
        if isinstance(drake_shape, DrakeBox):
            geometry = PydrakeToCollisionGeometryFactory.convert_box(
                drake_shape, represent_geometry_as, learnable)
        elif isinstance(drake_shape, DrakeHalfSpace):
            geometry = PydrakeToCollisionGeometryFactory.convert_plane()
        elif isinstance(drake_shape, DrakeMesh):
            geometry = PydrakeToCollisionGeometryFactory.convert_mesh(
                drake_shape, represent_geometry_as, learnable)
        elif isinstance(drake_shape, DrakeSphere):
            geometry = PydrakeToCollisionGeometryFactory.convert_sphere(
                drake_shape, represent_geometry_as, learnable)
        else:
            raise TypeError(
                "Unsupported type for drake Shape() to"
                "CollisionGeometry() conversion:", type(drake_shape))

        geometry.name = name
        return geometry

    @staticmethod
    def convert_box(drake_box: DrakeBox, represent_geometry_as: str,
                    learnable: bool = True
        ) -> Union[Box, Polygon]:
        """Converts ``pydrake.geometry.Box`` to ``Box`` or ``Polygon``."""
        if not learnable:
            represent_geometry_as = 'box'

        if represent_geometry_as == 'box':
            half_widths = 0.5 * Tensor(np.copy(drake_box.size()))
            return Box(half_widths, 4, learnable)

        if represent_geometry_as == 'polygon':
            pass # TODO

        raise NotImplementedError(f'Cannot presently represent a DrakeBox()' + \
            f'as {represent_geometry_as} type.')

    @staticmethod
    def convert_sphere(drake_sphere: DrakeSphere, represent_geometry_as: str,
                       learnable: bool = True
        ) -> Union[Sphere, Polygon]:
        """Converts ``pydrake.geometry.Sphere`` to ``Sphere``."""
        if not learnable:
            represent_geometry_as = 'sphere'

        if represent_geometry_as == 'sphere':
            return Sphere(torch.tensor([drake_sphere.radius()]), learnable)

        if represent_geometry_as == 'polygon':
            pass # TODO

        raise NotImplementedError(
            f'Cannot presently represent a DrakeSphere() as ' + \
            f'{represent_geometry_as} type.')

    @staticmethod
    def convert_plane() -> Plane:
        """Converts ``pydrake.geometry.HalfSpace`` to ``Plane``."""
        return Plane()

    @staticmethod
    def convert_mesh(
        drake_mesh: DrakeMesh, represent_geometry_as: str,
        learnable: bool = True) -> Union[DeepSupportConvex, Polygon]:
        """Converts ``pydrake.geometry.Mesh`` to ``Polygon`` or
        ``DeepSupportConvex``."""
        filename = drake_mesh.filename()
        mesh = pywavefront.Wavefront(filename)
        vertices = Tensor(mesh.vertices)

        if represent_geometry_as == 'mesh':
            return DeepSupportConvex(vertices, learnable=learnable)

        if represent_geometry_as == 'polygon':
            return Polygon(vertices, learnable=learnable)

        raise NotImplementedError(f'Cannot presently represent a ' + \
            f'DrakeMesh() as {represent_geometry_as} type.')


class GeometryCollider:
    """Utility class for colliding two ``CollisionGeometry`` instances."""

    @staticmethod
    def collide(geometry_a: CollisionGeometry, geometry_b: CollisionGeometry,
                R_AB: Tensor, p_AoBo_A: Tensor, vis_hook: None) -> \
            Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Collides two collision geometries.

        Takes in the two geometries as well as a relative transform between
        them. This function is a thin shell for other static methods of
        ``GeometryCollider`` where the given geometries are guaranteed to
        have specific types.

        Args:
            geometry_a: first collision geometry
            geometry_b: second collision geometry, with type
              ordering ``not geometry_A > geometry_B``.
            R_AB: (\*, 3, 3) rotation between geometry frames
            p_AoBo_A: (\*, 3) offset of geometry frame origins

        Returns:
            (\*, N) batch of witness point pair distances
            (\*, N, 3, 3) contact frame C rotation in A, R_AC, where the z
            axis of C is contained in the normal cone of body A at contact
            point Ac and is parallel (or antiparallel) to AcBc.
            (\*, N, 3) witness points Ac on A, p_AoAc_A
            (\*, N, 3) witness points Bc on B, p_BoBc_B
        """
        assert not geometry_a > geometry_b

        # case 1: half-space to compact-convex collision
        if isinstance(geometry_a, Plane) and isinstance(
                geometry_b, BoundedConvexCollisionGeometry):
            return GeometryCollider.collide_plane_convex(
                geometry_b, R_AB, p_AoBo_A, vis_hook)
        if isinstance(geometry_a, Sphere) and isinstance(
                geometry_b, SparseVertexConvexCollisionGeometry):
            if DEBUG_TRIMESH_COLLISIONS:
                return GeometryCollider.collide_sphere_sparse_convex(
                    geometry_a, geometry_b, R_AB, p_AoBo_A)
            else:
                return GeometryCollider.collide_sphere_sparse_convex_parallel(
                    geometry_a, geometry_b, R_AB, p_AoBo_A, vis_hook)
        if isinstance(geometry_a, BoundedConvexCollisionGeometry) and \
            isinstance(geometry_b, BoundedConvexCollisionGeometry):
            return GeometryCollider.collide_convex_convex(
                geometry_a, geometry_b, R_AB, p_AoBo_A)
        raise TypeError(
            "No type-specific implementation for geometry "
            "pair of following types:",
            type(geometry_a).__name__,
            type(geometry_b).__name__)

    @staticmethod
    def collide_plane_convex(geometry_b: BoundedConvexCollisionGeometry,
                             R_AB: Tensor, p_AoBo_A: Tensor,
                             vis_hook: None) -> \
            Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Implementation of ``GeometryCollider.collide()`` when
        ``geometry_a`` is a ``Plane`` and ``geometry_b`` is a
        ``BoundedConvexCollisionGeometry``."""
        R_BA = R_AB.transpose(-1, -2)

        # support direction on body B is negative z axis of A frame,
        # in B frame coordinates, i.e. the final column of ``R_BA``.
        directions_b = -R_BA[..., 2]

        # B support points of shape (*, N, 3)
        p_BoBc_B = geometry_b.support_points(directions_b)
        p_BoBc_B.register_hook(vis_hook.hook_grad_plane_and_object)
        p_AoBc_A = pbmm(p_BoBc_B, R_BA) + p_AoBo_A.unsqueeze(-2)

        # phi is the A-axes z coordinate of Bc
        phi = p_AoBc_A[..., 2]

        # ### phi from network first order forwarding
        # phi_net_BoBc_B = geometry_b.network.get_output(directions_b)
        # phi_net = p_AoBo_A[:,2] - phi_net_BoBc_B
        # phi = phi_net.unsqueeze(-1).expand_as(phi)

        # Ac is the projection of Bc onto the z=0 plane in frame A.
        # pylint: disable=E1103
        p_AoAc_A = torch.cat(
            (p_AoBc_A[..., :2], torch.zeros_like(p_AoBc_A[..., 2:])), -1)

        # ``R_AC`` (\*, N, 3, 3) is simply a batch of identities, as the z
        # axis of A points out of the plane.
        # pylint: disable=E1103
        R_AC = torch.eye(3).expand(p_AoAc_A.shape + (3,))

        return phi, R_AC, p_AoAc_A, p_BoBc_B

    @staticmethod
    def collide_sphere_sparse_convex(
        geometry_a: Sphere,
        geometry_b: SparseVertexConvexCollisionGeometry,
        R_AB: Tensor,
        p_AoBo_A: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Custom implementation of ``GeometryCollider.collide()`` when the
        first geometry is a sphere and the second geometry is a sparse vertex
        convex collision geometry.

        NOTE:  This implementation is not parallelized and was developed / can
        be used for debugging purposes.  Otherwise it is unused in favor of
        ``collide_sphere_sparse_convex_parallel``.

        Args:
            R_AB: (\*, 3, 3) rotation between geometry frames
            p_AoBo_A: (\*, 3) offset of geometry frame origins

        Returns:
            (\*, N) batch of witness point pair distances
            (\*, N, 3, 3) contact frame C rotation in A, R_AC, where the z
            axis of C is contained in the normal cone of body A at contact
            point Ac and is parallel (or antiparallel) to AcBc.
            (\*, N, 3) witness points Ac on A, p_AoAc_A
            (\*, N, 3) witness points Bc on B, p_BoBc_B
        """
        # Call network directly for DeepSupportConvex objects.
        support_fn_a = geometry_a.support_points
        support_fn_b = geometry_b.support_points
        if isinstance(geometry_b, DeepSupportConvex):
            support_fn_b = geometry_b.network

        ORIGIN_XYZ = np.array([0,0,0]).reshape(1, 3)

        # Get shapes of inputs, ensuring of correct dimensions.
        p_AoBo_A = p_AoBo_A.unsqueeze(-2)
        original_batch_dims = p_AoBo_A.shape[:-2]
        p_AoBo_A = p_AoBo_A.view(-1, 3)
        R_AB = R_AB.view(-1, 3, 3)
        batch_range = p_AoBo_A.shape[0]

        # Get the vertex set of the second geometry and define a trimesh object
        # from it.
        directions = torch.zeros_like(p_AoBo_A)
        b_vertices_B = geometry_b.get_vertices(
            directions, sample_entire_mesh=True)
        trimesh_mesh = trimesh.Trimesh(
            vertices=b_vertices_B[0].detach().numpy()).convex_hull
        trimesh_mesh.process()

        for transform_index in range(batch_range):
            # Transform the mesh.
            T_AB = np.eye(4)
            T_AB[:3, :3] = R_AB[transform_index].detach().numpy()
            T_AB[:3, 3] = p_AoBo_A[transform_index].detach().numpy()
            trimesh_mesh.apply_transform(T_AB)

            # Find nearest points.
            closest_points, _distances, _triangle_ids = \
                trimesh.proximity.closest_point(trimesh_mesh, ORIGIN_XYZ)
            closest_point = closest_points[0]

            # Rely on Trimesh's signed # distance function.  NOTE: Negative sign
            # here because Trimesh uses points inside the mesh have positive
            # signed distance, which is opposite of our convention.
            signed_distance = -trimesh.proximity.signed_distance(
                trimesh_mesh, ORIGIN_XYZ)[0] - geometry_a.get_radius()

            # Regardless of whether a collision has occurred, the (unscaled)
            # direction of the contact is the closest point on the mesh, since
            # the sphere is located at the origin.  This direction has to be
            # flipped if the origin is inside the mesh.
            directions[transform_index] += closest_point
            if signed_distance < -geometry_a.get_radius():
                directions[transform_index] *= -1

            if DEBUG_TRIMESH_COLLISIONS:
                def set_transparency(mesh, alpha):
                    n_vertices = len(mesh.vertices)
                    colors = np.ones((n_vertices, 4)) * 255
                    colors[:, 3] = alpha * 255
                    mesh.visual.vertex_colors = colors

                '''Debugging visuals:
                 - Red dot:  closest point on mesh.
                 - Cyan dot:  point on mesh generated from query direction.
                 - Blue/chartreuse dot:  closest point on sphere (will be on top
                    of dot generated from query direction, so only chartreuse
                    will be visible).
                 - Dark green dots:  query directions for each object.
                '''

                # Show sphere and mesh geometries.
                trimesh_sphere = trimesh.creation.icosphere(
                    radius=geometry_a.get_radius().item())
                set_transparency(trimesh_sphere, 0.5)
                set_transparency(trimesh_mesh, 0.5)

                # Label the trimesh-generated witness points.
                mesh_witness_pt = closest_point
                sphere_witness_pt = closest_point/np.linalg.norm(closest_point)\
                    * geometry_a.get_radius().item()

                trimesh_mesh_witness = trimesh.points.PointCloud(
                    mesh_witness_pt.reshape(1, 3),
                    colors=np.array([[255, 0, 0, 255]]))
                trimesh_sphere_witness = trimesh.points.PointCloud(
                    sphere_witness_pt.reshape(1, 3),
                    colors=np.array([[0, 0, 255, 255]]))

                # Show triad for each object.
                triad_A = trimesh.creation.axis(
                    origin_size=0, transform=np.eye(4), axis_radius=0.0025,
                    axis_length=0.05)
                triad_B = trimesh.creation.axis(
                    origin_size=0, transform=T_AB, axis_radius=0.0025,
                    axis_length=0.05)

                # Get query directions and points.
                direction_A = directions[transform_index]
                direction_A = direction_A / np.linalg.norm(direction_A)
                direction_B = -pbmm(direction_A.unsqueeze(0),
                                    R_AB[transform_index]).squeeze(0)
                p_AoAc_A = support_fn_a(direction_A).squeeze(0)
                p_BoBc_B = support_fn_b(direction_B).squeeze(0)
                p_BoBc_A = pbmm(p_BoBc_B.unsqueeze(0),
                                R_AB[transform_index].transpose(-1, -2)
                                ).squeeze(0)
                p_AcBc_A = -p_AoAc_A + p_AoBo_A[transform_index] + p_BoBc_A
                p_AoBc_A = p_AoBo_A[transform_index] + p_BoBc_A

                # Label the support function generated witness points.
                support_sphere_witness = trimesh.points.PointCloud(
                    p_AoAc_A.detach().numpy().reshape(1, 3),
                    colors=np.array([[255, 255, 0, 255]]))
                support_mesh_witness = trimesh.points.PointCloud(
                    p_AoBc_A.detach().numpy().reshape(1, 3),
                    colors=np.array([[0, 255, 255, 255]]))

                # Show the query direction for each object, starting from their
                # origins.
                direction_A_np = direction_A.detach().numpy()
                scalars = np.linspace(0, 0.05, 10)
                ps_AoQueryA_A = scalars[:, np.newaxis] * direction_A_np
                ps_BoQueryB_A = T_AB[:3, 3] - ps_AoQueryA_A
                trimesh_query_A = trimesh.points.PointCloud(
                    ps_AoQueryA_A, colors=np.array([[0, 255, 0, 255]]))
                trimesh_query_B = trimesh.points.PointCloud(
                    ps_BoQueryB_A, colors=np.array([[0, 255, 0, 255]]))


                # Print debugging aids.
                R_AC = rotation_matrix_from_one_vector(
                    directions[transform_index], 2)
                phi = (p_AcBc_A * R_AC[..., 2]).sum(dim=-1)
                trimesh_phi = signed_distance
                print(f'{T_AB=}')
                print(f'{direction_A=}')
                print(f'{direction_B=}')
                print(f'{p_AoAc_A=}')
                print(f'{p_BoBc_B=}')
                print(f'{p_BoBc_A=}')
                print(f'{p_AcBc_A=}')
                print(f'{R_AC=}')
                print(f'{phi=}')
                print(f'{trimesh_phi=}')
                # pdb.set_trace()

                scene = trimesh.Scene([
                    trimesh_mesh, trimesh_sphere,
                    trimesh_mesh_witness, trimesh_sphere_witness,
                    triad_A, triad_B, trimesh_query_A, trimesh_query_B,
                    support_mesh_witness, support_sphere_witness
                ])
                scene.show()

            # Undo the transformation in preparation for the next
            # transformation.
            trimesh_mesh.apply_transform(np.linalg.inv(T_AB))

        # Get normal directions in each object frame.
        directions_A = directions / directions.norm(dim=-1, keepdim=True)
        directions_B = -pbmm(directions_A.unsqueeze(-2), R_AB).squeeze(-2)

        p_AoAc_A = support_fn_a(directions_A).squeeze(-2)
        p_BoBc_B = support_fn_b(directions_B)
        p_BoBc_A = pbmm(p_BoBc_B.unsqueeze(-2),
                        R_AB.transpose(-1, -2)).squeeze(-2)

        p_AcBc_A = -p_AoAc_A + p_AoBo_A + p_BoBc_A

        # Get phi as the projection of the vector between contact points along
        # the contact normal direction.
        R_AC = rotation_matrix_from_one_vector(directions, 2)
        phi = (p_AcBc_A * R_AC[..., 2]).sum(dim=-1)

        phi = phi.reshape(original_batch_dims + (1,))
        R_AC = R_AC.reshape(original_batch_dims + (1, 3, 3))
        p_AoAc_A = p_AoAc_A.reshape(original_batch_dims + (1, 3))
        p_BoBc_B = p_BoBc_B.reshape(original_batch_dims + (1, 3))
        return phi, R_AC, p_AoAc_A, p_BoBc_B

    @staticmethod
    def collide_sphere_sparse_convex_parallel(
        geometry_a: Sphere,
        geometry_b: SparseVertexConvexCollisionGeometry,
        R_AB: Tensor,
        p_AoBo_A: Tensor,
        vis_hook: None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Implementation of ``GeometryCollider.collide()`` when the first
        geometry is a sphere and the second geometry is a sparse vertex convex
        collision geometry.  Leverages trimesh for signed distance queries.

        This is a parallelized version of ``collide_sphere_sparse_convex`` that
        seems to have similar quality of results while running anywhere from
        2.5x to 10x faster."""
        # Call network directly for DeepSupportConvex objects.
        support_fn_a = geometry_a.support_points
        support_fn_b = geometry_b.support_points
        if isinstance(geometry_b, DeepSupportConvex):
            support_fn_b = geometry_b.network

        # Get shapes of inputs, ensuring of correct dimensions.
        p_AoBo_A = p_AoBo_A.unsqueeze(-2)
        original_batch_dims = p_AoBo_A.shape[:-2]
        p_AoBo_A = p_AoBo_A.view(-1, 3)
        R_AB = R_AB.view(-1, 3, 3)

        # Fix the geometry B origin, and compute new locations for A expressed
        # in B frame.  This allows us to query the mesh at the same location for
        # multiple relative sphere locations at a time, speeding up the signed
        # distance queries.
        p_BoAo_B = -pbmm(p_AoBo_A.unsqueeze(-2), R_AB).squeeze(-2)
        R_BA = R_AB.transpose(-1, -2)

        # Get the vertex set of the second geometry and define a trimesh object
        # from it.
        directions_A_to_B_in_B = torch.zeros_like(p_BoAo_B)
        b_vertices_B = geometry_b.get_vertices(
            directions_A_to_B_in_B, sample_entire_mesh=True)
        trimesh_mesh = trimesh.Trimesh(
            vertices=b_vertices_B[0].detach().numpy()).convex_hull
        trimesh_mesh.process()

        # Find nearest points on mesh to the centers of the sphere at its
        # various locations.
        closest_points, _distances, _triangle_ids = \
            trimesh.proximity.closest_point(trimesh_mesh, p_BoAo_B)
        # closest_points.shape == (*, 3)

        # Query the signed distances from the mesh to the sphere centers.
        # NOTE: Negative sign here because trimesh uses points inside the mesh
        # have positive signed distance, which is opposite of our convention.
        signed_distances = -trimesh.proximity.signed_distance(
            trimesh_mesh, p_BoAo_B) - geometry_a.get_radius().item()

        # Use the vector from the sphere center to the closest point on the mesh
        # as the contact direction.  If the sphere's origin is inside the mesh,
        # flip the direction to ensure it always points from the sphere into the
        # mesh.
        directions_A_to_B_in_B = closest_points - p_BoAo_B.detach().numpy()
        to_flip_mask = signed_distances < -geometry_a.get_radius().item()
        directions_A_to_B_in_B[to_flip_mask] *= -1

        directions_A_to_B_in_B = Tensor(directions_A_to_B_in_B)

        directions_A_to_B_in_A = pbmm(
            directions_A_to_B_in_B.unsqueeze(-2), R_BA).squeeze(-2)
        R_AC = rotation_matrix_from_one_vector(directions_A_to_B_in_A, 2)
        # R_AC[..., 2] is normalized directions_A_to_B_in_A
        # Get normal directions in each object frame.
        directions_B = -directions_A_to_B_in_B / directions_A_to_B_in_B.norm(
            dim=-1, keepdim=True)
        directions_A = -pbmm(directions_B.unsqueeze(-2), R_BA).squeeze(-2)

        p_AoAc_A = support_fn_a(directions_A).squeeze(-2)
        p_BoBc_B = support_fn_b(directions_B)
        vis_hook.record('directions_B', directions_B)
        p_BoBc_B.register_hook(vis_hook.hook_grad_sphere_and_object)
        p_BoBc_A = pbmm(p_BoBc_B.unsqueeze(-2),
                        R_AB.transpose(-1, -2)).squeeze(-2)

        p_AcBc_A = -p_AoAc_A + p_AoBo_A + p_BoBc_A
        ### Check if the gradient wrt p_BoBc_B from comp loss is correct. 
        # vis_hook.record('directions_A_to_B_in_A', directions_A_to_B_in_A)
        # vis_hook.record('directions_A_to_B_in_B', directions_A_to_B_in_B)
        # p_BoBc_B.register_hook(lambda grad: vis_hook.hook_check_grad('p_BoBc_B', grad))
        # p_BoBc_A.register_hook(lambda grad: vis_hook.hook_check_grad('p_BoBc_A', grad))
        # p_AcBc_A.register_hook(lambda grad: vis_hook.hook_check_grad('p_AcBc_A', grad))
        phi = (p_AcBc_A * R_AC[..., 2]).sum(dim=-1)

        phi = phi.reshape(original_batch_dims + (1,))
        R_AC = R_AC.reshape(original_batch_dims + (1, 3, 3))
        p_AoAc_A = p_AoAc_A.reshape(original_batch_dims + (1, 3))
        p_BoBc_B = p_BoBc_B.reshape(original_batch_dims + (1, 3))
        return phi, R_AC, p_AoAc_A, p_BoBc_B

    @staticmethod
    def collide_convex_convex(
            geometry_a: BoundedConvexCollisionGeometry,
            geometry_b: BoundedConvexCollisionGeometry,
            R_AB: Tensor,
            p_AoBo_A: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Implementation of ``GeometryCollider.collide()`` when
        both geometries are ``BoundedConvexCollisionGeometry``\es.

        CAUTION:  FCL was found to be unreliable in some cases.  Some of the
        issues were addressed with ``fcl_direction_cleaner`` and
        ``fcl_distance_nearest_points_cleaner``, but there are other observed
        issues in the collision detection and the associated contact normals.
        """
        raise RuntimeError(
            "This function is not currently supported due to issues with FCL.")
        # pylint: disable=too-many-locals

        # Call network directly for DeepSupportConvex objects
        support_fn_a = geometry_a.support_points
        support_fn_b = geometry_b.support_points
        if isinstance(geometry_a, DeepSupportConvex):
            support_fn_a = geometry_a.network
        if isinstance(geometry_b, DeepSupportConvex):
            support_fn_b = geometry_b.network

        p_AoBo_A = p_AoBo_A.unsqueeze(-2)
        original_batch_dims = p_AoBo_A.shape[:-2]
        p_AoBo_A = p_AoBo_A.view(-1, 3)
        R_AB = R_AB.view(-1, 3, 3)
        batch_range = p_AoBo_A.shape[0]

        # Assume collision directions are piecewise constant, which allows us
        # to use :py:mod:`fcl` to compute the direction, without the need to
        # differentiate through it.
        # pylint: disable=E1103
        directions = torch.zeros_like(p_AoBo_A)

        # Used if there are multiple coplanar witness points
        # To determine the actual contact point in a differentiable manner
        hints_a = torch.zeros_like(p_AoBo_A)
        hints_b = torch.zeros_like(p_AoBo_A)

        # setup fcl=
        a_fcl_geometry = geometry_a.get_fcl_geometry()
        b_fcl_geometry = geometry_b.get_fcl_geometry()
        a_obj = fcl.CollisionObject(a_fcl_geometry, fcl.Transform())
        b_obj = fcl.CollisionObject(b_fcl_geometry, fcl.Transform())

        collision_request = fcl.CollisionRequest()
        collision_request.enable_contact = True
        distance_request = fcl.DistanceRequest()
        distance_request.enable_nearest_points = True

        for transform_index in range(batch_range):
            b_t = fcl.Transform(R_AB[transform_index].detach().numpy(),
                                p_AoBo_A[transform_index].detach().numpy())
            b_obj.setTransform(b_t)
            result = fcl.CollisionResult()
            if fcl.collide(a_obj, b_obj, collision_request, result) > 0:
                # Collision detected.
                # Assume only 1 contact point.
                directions[transform_index] += fcl_collision_direction_cleaner(
                    a_fcl_geometry, b_fcl_geometry, result)
                nearest_points = [
                    result.contacts[0].pos + \
                        result.contacts[0].penetration_depth/2.0 * \
                        result.contacts[0].normal,
                    result.contacts[0].pos - \
                        result.contacts[0].penetration_depth/2.0 * \
                        result.contacts[0].normal
                ]
            else:
                result = fcl.DistanceResult()
                a_pt_A, b_pt_A = fcl_distance_nearest_points_cleaner(
                    a_fcl_geometry, b_fcl_geometry, a_obj, b_obj,
                    distance_request, result)
                directions[transform_index] += b_pt_A - a_pt_A

                nearest_points = (a_pt_A, b_pt_A)

            # Record Hints == expected contact point in each object's frame
            hints_a[transform_index] = torch.tensor(nearest_points[0])
            hints_b[transform_index] = pbmm(
                torch.tensor(nearest_points[1]) - \
                    p_AoBo_A[transform_index].detach(),
                R_AB[transform_index]
            )

        R_AC = rotation_matrix_from_one_vector(directions, 2)

        # Get normal directions in each object frame
        directions_A = directions / directions.norm(dim=-1, keepdim=True)
        directions_B = -pbmm(directions_A.unsqueeze(-2), R_AB).squeeze(-2)

        p_AoAc_A = support_fn_a(directions_A, hints_a).squeeze(-2)
        p_BoBc_B = support_fn_b(directions_B, hints_b)
        p_BoBc_A = pbmm(p_BoBc_B.unsqueeze(-2),
                        R_AB.transpose(-1, -2)).squeeze(-2)

        p_AcBc_A = -p_AoAc_A + p_AoBo_A + p_BoBc_A

        phi = (p_AcBc_A * R_AC[..., 2]).sum(dim=-1)

        phi = phi.reshape(original_batch_dims + (1,))
        R_AC = R_AC.reshape(original_batch_dims + (1, 3, 3))
        p_AoAc_A = p_AoAc_A.reshape(original_batch_dims + (1, 3))
        p_BoBc_B = p_BoBc_B.reshape(original_batch_dims + (1, 3))
        return phi, R_AC, p_AoAc_A, p_BoBc_B


def fcl_collision_direction_cleaner(
        a_geom: fcl.CollisionGeometry,
        b_geom: fcl.CollisionGeometry,
        result: fcl.CollisionResult) -> Tuple[Tensor, Tensor]:
    """Compensates for bugs in FCL's collision reults.  The primary issue is
    that it can swap the order of the objects in the result, meaning the normal
    direction is also swapped.  This function detects when this is the case,
    corrects the result, and returns the processed direction vector.

    NOTE:  This assumes the result was already used in a collision request i.e.
    ``fcl.collide()`` was called.

    NOTE:  This assumes a single contact by always grabbing the 0th index
    result.

    Args:
        a_geom: first collision geometry
        b_geom: second collision geometry
        result: collision result

    Returns:
        (3,) contact normal vector from A to B
    """
    # Detect if the order of the result's objects matches the input arguments.
    if result.contacts[0].o1 == a_geom:
        assert result.contacts[0].o2 == b_geom
        normal_direction = Tensor(result.contacts[0].normal)

    else:
        assert result.contacts[0].o1 == b_geom
        assert result.contacts[0].o2 == a_geom
        normal_direction = -Tensor(result.contacts[0].normal)

    return normal_direction


def fcl_distance_nearest_points_cleaner(
        a_geom: fcl.CollisionGeometry,
        b_geom: fcl.CollisionGeometry,
        a_obj: fcl.CollisionObject,
        b_obj: fcl.CollisionObject,
        distance_request: fcl.DistanceRequest,
        result: fcl.DistanceResult) -> Tuple[Tensor, Tensor]:
    """Compensates for bugs in FCL's distance reults.  The primary issue is that
    while the distance reported by the call to ``fcl.distance()`` is correct,
    the nearest points can be incorrect.  Through experimentation, it was
    discovered that usually this issue is due to the nearest points being
    expressed in different frames.  This function detects when this is the case,
    corrects the result, and returns the processed nearest points both expressed
    in object A frame.

    Args:
        a_geom: first collision geometry
        b_geom: second collision geometry
        a_obj: first collision object, containing a_geom as its geometry
        b_obj: second collision object, containing b_geom as its geometry
        distance_request: distance request
        result: distance result

    Returns:
        (3,) p_AoAc_A
        (3,) p_AoBc_A
    """
    # Threshold for checking the distance result conversion worked.
    eps = 1e-5      # 1e-5 is a hundredth of a millimeter.

    # Compute the distance request.
    distance = fcl.distance(a_obj, b_obj, distance_request, result)

    # Determine the order of the objects in the result.
    a_index = 0 if result.o1 == a_geom else 1
    b_index = 0 if result.o1 == b_geom else 1
    assert a_index != b_index
    if a_index == 0:
        assert result.o1 == a_geom
        assert result.o2 == b_geom
        flipped = False
    else:
        assert result.o1 == b_geom
        assert result.o2 == a_geom
        flipped = True

    # Ensure the nearest points make sense.
    if not flipped:
        a_pt_A = Tensor(result.nearest_points[a_index])
        b_pt_A = Tensor(result.nearest_points[b_index])
    else:
        a_pt_A = Tensor(result.nearest_points[a_index])
        b_pt_B = Tensor(result.nearest_points[b_index])

        assert np.all(a_obj.getQuatRotation() == np.array([1, 0, 0, 0]))
        assert np.all(a_obj.getTranslation() == np.array([0, 0, 0]))

        p_AoBo_A = Tensor(b_obj.getTranslation())
        R_AB = Tensor(b_obj.getRotation())

        b_pt_A = pbmm(b_pt_B, R_AB.transpose(-1, -2)) + p_AoBo_A

    assert torch.abs(distance - torch.linalg.norm(b_pt_A - a_pt_A)).item() < eps

    return a_pt_A, b_pt_A
