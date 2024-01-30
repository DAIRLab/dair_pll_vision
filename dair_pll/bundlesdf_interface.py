"""Functionality related to connecting ContactNets to BundleSDF."""

import os
import os.path as op
import pdb
from typing import Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Path3DCollection
import numpy as np
from dair_pll.system import MeshSummary
import torch
from torch import Tensor
from scipy.spatial import ConvexHull  # type: ignore

from dair_pll import deep_support_function, file_utils
from dair_pll.geometry import _DEEP_SUPPORT_DEFAULT_DEPTH, \
    _DEEP_SUPPORT_DEFAULT_WIDTH
from dair_pll.deep_support_function import HomogeneousICNN
from dair_pll.file_utils import EXPORT_POINTS_DEFAULT_NAME, \
    EXPORT_DIRECTIONS_DEFAULT_NAME, \
    EXPORT_FORCES_DEFAULT_NAME


TEST_RUN_NAME = 'test_004'
SYSTEM_NAME = 'bundlesdf_cube'

# Hyperparameters for querying into and out of the object at an SDF=0 point.
N_QUERY_INSIDE = 100
N_QUERY_OUTSIDE = 50
N_QUERY_OUTSIDE_FAR = 50
DEPTH_INSIDE = 0.005
DEPTH_OUTSIDE = 0.005
DEPTH_FAR_OUTSIDE = 0.1

# Amended values for the above hyperparameters for when sampling from mesh.  The
# total number of queries per point is lower since more points are expected to
# be sampled, and sticking closer to the surface of the mesh.
MESH_N_QUERY_INSIDE = 10
MESH_N_QUERY_OUTSIDE = 10
MESH_N_QUERY_OUTSIDE_FAR = 0
MESH_DEPTH_INSIDE = 0.02
MESH_DEPTH_OUTSIDE = 0.02
MESH_DEPTH_FAR_OUTSIDE = 0.1
N_MESH_SAMPLE = 25000

# Amended values for the above hyperparameters for when sampling from mesh to
# obtain points for enforcing monotonic SDF increase away from object.
GRADIENT_N_QUERY_INSIDE = 4
GRADIENT_N_QUERY_OUTSIDE = 10
GRADIENT_N_QUERY_OUTSIDE_FAR = 10
GRADIENT_DEPTH_INSIDE = 0.005
GRADIENT_DEPTH_OUTSIDE = 0.02
GRADIENT_DEPTH_FAR_OUTSIDE = 0.1

# Hyperparameters for querying around an object with SDF minimum bounds.
BOUNDED_NEARBY_DEPTH = 0.005
BOUNDED_NEARBY_RADIUS = 0.05
BOUNDED_FAR_DEPTH = 0.1
BOUNDED_FAR_RADIUS = 0.1
BOUNDED_FAR_N_QUERY = 100
BOUNDED_NEARBY_OUTSIDE_N_QUERY = 100
# Make the inside queried points equal to all queried outside.
BOUNDED_NEARBY_INSIDE_N_QUERY = BOUNDED_FAR_N_QUERY + \
    BOUNDED_NEARBY_OUTSIDE_N_QUERY

# Hyperparameters for filtering support points or hull sample points
FORCE_THRESH = 0.3676                       # Newtons
HULL_PROXIMITY_THRESH = 0.001               # meters
SUPPORT_POINT_DISTANCE_THRESHOLD = 0.010    # meters

# Flags for running some unit tests.
DO_SMALL_FILTERING_AND_VISUALIZATION_TEST = False
DO_SDFS_FROM_MESH_SAMPLING_WITH_SUPPORT_FILTERING = False
DO_COMBINE_SUPPORT_POINTS_AND_MESH_SAMPLING = False
DO_NETWORK_LOADING_TEST = False
DO_GRADIENT_DATA_TEST = False
DO_UNIQUENESS_SCORE_TEST = False
DO_DOUBLE_UNIQUENESS_SCORE_TEST = False
DO_SUPPORT_POINT_SNAPPING_TEST = False


# ========================= Data Generation Helpers ========================== #
def filter_points_and_directions_based_on_contact(
        contact_points: Tensor, directions: Tensor, normal_forces: Tensor,
        force_threshold: float = FORCE_THRESH) -> Tuple[Tensor, Tensor]:
    """Filter out points that are likely not in contact with the ground during
    the data captured by the normal_forces tensor.

    Args:
        contact_points (M, 3):  M support points of the object geometry for the
            given support directions.
        directions (M, 3):  associated support directions.
        normal_forces (M,):  normal forces associated with the contact points
            during one timestep of PLL training.
        force_threshold:  force threshold below which contact is decided to not
            have occurred at a support point.

    Outputs:
        filtered_points (N, 3):  N support points that experienced a normal
            force greater than force_threshold.
        filtered_directions (N, 3):  the corresponding normal directions.
    """
    assert normal_forces.ndim == 1
    assert contact_points.ndim == directions.ndim == 2
    assert normal_forces.shape[0]==contact_points.shape[0]==directions.shape[0]
    assert contact_points.shape[1] == directions.shape[1] == 3

    mask = normal_forces > force_threshold
    filtered_points = contact_points[mask]
    filtered_directions = directions[mask]

    return filtered_points.detach(), filtered_directions.detach()


def filter_mesh_samples_based_on_supports(
        sample_points: Tensor, sample_normals: Tensor, contact_points: Tensor,
        support_directions: Tensor, threshold: float = HULL_PROXIMITY_THRESH
) -> Tuple[Tensor, Tensor]:
    """Given a set of points sampled on the convex hull mesh and their outward
    normal directions, filter out any that are located beyond a threshold away
    from support point hyperplanes.

    Args:
        sample_points (N, 3)
        sample_normals (N, 3)
        contact_points (M, 3)
        support_directions (M, 3)
        threshold:  maximum distance threshold in meters permissible between 
            sample point and the nearest hyperplane defined by contact_points
            and support_directions.

    Outputs:
        filtered_sample_points (K, 3)
        filtered_sample_directions (K, 3)
    """
    # Do some input checking.
    assert sample_points.shape == sample_normals.shape
    assert contact_points.shape == support_directions.shape
    assert sample_points.ndim == contact_points.ndim == 2
    assert sample_points.shape[1] == contact_points.shape[1] == 3

    # Use 3D arrays with indexing [sample_i, support_i, x/y/z].
    n_samples = sample_points.shape[0]
    n_supports = contact_points.shape[0]
    sample_points_expanded = sample_points.unsqueeze(1).repeat(1, n_supports, 1)
    support_points_expanded = contact_points.unsqueeze(0).repeat(n_samples, 1, 1)
    support_dirs_expanded = support_directions.unsqueeze(0).repeat(n_samples, 1, 1)

    # Get distances from hyperplanes by projecting sample points onto hyperplane
    # normal axis.
    support_to_sample_vec = sample_points_expanded - support_points_expanded
    hyperplane_distances = torch.einsum('ijk,ijk->ij',
                                        support_to_sample_vec,
                                        support_dirs_expanded
                                        )   # (n_samples, n_supports)
    
    # Samples should be kept if their minimum distance to the hyperplanes is
    # less than the threshold.
    within_threshold = torch.abs(hyperplane_distances) < threshold
    sample_mask = torch.sum(within_threshold, dim=1) > 0

    # Return the filtered sample points and directions.
    return sample_points[sample_mask], sample_normals[sample_mask]


def create_mesh_from_deep_support(deep_support: HomogeneousICNN) -> MeshSummary:
    """Create a mesh from a deep support convex network.

    Args:
        deep_support:  a deep support convex network.

    Outputs:
        A MeshSummary with vertices and faces attributes.  The vertices are
            already listed in counter-clockwise order.
    """
    return deep_support_function.extract_mesh(deep_support)


def create_mesh_from_set_of_points(points: Tensor) -> MeshSummary:
    """Given a set of points, extracts a vertex/face mesh.

    Args:
        points (N, 3):  set of 3D points.

    Returns:
        A mesh summary.
    """
    support_points = points
    support_point_hashes = set()
    unique_support_points = []

    # remove duplicate vertices
    for vertex in support_points:
        vertex_hash = hash(vertex.numpy().tobytes())
        if vertex_hash in support_point_hashes:
            continue
        support_point_hashes.add(vertex_hash)
        unique_support_points.append(vertex)

    vertices = torch.stack(unique_support_points)
    hull = ConvexHull(vertices.numpy())
    faces = Tensor(hull.simplices).to(torch.long)  # type: ignore

    _, backwards, _ = deep_support_function.extract_outward_normal_hyperplanes(
        vertices.unsqueeze(0), faces.unsqueeze(0))
    backwards = backwards.squeeze(0)
    faces[backwards] = faces[backwards].flip(-1)

    return MeshSummary(vertices=vertices, faces=faces)


def sample_on_mesh(mesh: MeshSummary, n_sample: int,
                   weighted_by_area: bool = True) -> Tuple[Tensor, Tensor]:
    """Sample points that are on the mesh and store their corresponding outward
    normal.

    Args:
        mesh:  a MeshSummary.
        n_sample:  number of points to sample.
        weighted_by_area:  whether to select a mesh triangle with probability
            proportional to its area or to give every mesh triangle the same
            likelihood of selection.

    Outputs:
        surface_points (N, 3):  points strictly on mesh surface.
        surface_normals (N, 3):  outward surface normals.
    """
    # Get a probability distribution.
    n_faces = mesh.faces.shape[0]
    probabilities = np.ones(n_faces) / n_faces
    if weighted_by_area:
        face_verts = mesh.vertices[mesh.faces]
        vecs_1 = face_verts[:, 1] - face_verts[:, 0]
        vecs_2 = face_verts[:, 2] - face_verts[:, 1]
        face_areas = torch.linalg.norm(torch.cross(vecs_1, vecs_2), dim=1) / 2
        probabilities = face_areas / torch.sum(face_areas)

    # Randomly select faces.  Index into vertices via [face_i, vertex_i, x/y/z].
    indices = torch.multinomial(probabilities, n_sample, replacement=True)
                                                            # (n_sample,)
    faces = mesh.faces[indices]                             # (n_sample, 3)
    vertices = mesh.vertices[faces]                         # (n_sample, 3, 3)

    # Store the faces' outward normals.
    all_surface_normals, _, _ = \
        deep_support_function.extract_outward_normal_hyperplanes(
            mesh.vertices, mesh.faces
        )
    surface_normals = all_surface_normals.squeeze(0)[indices]

    # Generate points randomly interpolated inside selected vertices.
    abcs = torch.rand((n_sample, 3))
    abcs /= torch.sum(abcs, dim=1).unsqueeze(1).tile(1,3)   # (n_sample, 3)
    abcs = abcs.unsqueeze(2).tile(1,1,3)                    # (n_sample, 3, 3)
    vert_pieces = vertices * abcs
    surface_points = torch.sum(vert_pieces, dim=1)          # (n_sample, 3)

    return surface_points, surface_normals
    

def compute_position_and_direction_uniquenesses(
        points: Tensor, directions: Tensor) -> Tuple[Tensor, Tensor]:
    """Compute scores for how "uniquely" each position and direction is
    represented in the provided points and directions arrays.  The higher the
    score, the more underrepresented the point's position or direction.  The
    scores physically mean:

        Position uniqueness is the average distance away a point is from all the
            other points.  This score is in [0, \infty) where 0 means the point
            is centered among all other points, and anything larger represents
            the distance in meters to the centroid of its neighbors.

        Direction uniqueness is the average distance away the tip of an origin-
            centered direction vector is from all the other direction vector
            ends.  This score is in [0, \infty) where 0 means the direction is
            centered among all other centers, and anything larger represents
            the distance away to the centroid of other direction vectors.

    Args:
        points (N, 3)
        directions (N, 3)

    Outputs:
        position_uniqueness (N,)
        direction_uniqueness (N,)
    """
    n_points = points.shape[0]
    n_neighbors = n_points - 1
    
    # Use 3D arrays with indexing [point_i, neighbor_i, x/y/z].
    points_expanded = points.unsqueeze(1).repeat(1, n_neighbors, 1)
    neighbors = points.unsqueeze(0).repeat(n_points, 1, 1)
    neighbors = neighbors[torch.eye(n_points) == 0]     # don't compare to self.
    neighbors = neighbors.reshape(n_points, n_neighbors, 3)

    directions_expanded = directions.unsqueeze(1).repeat(1, n_neighbors, 1)
    neighbor_dirs = directions.unsqueeze(0).repeat(n_points, 1, 1)
    neighbor_dirs = neighbor_dirs[torch.eye(n_points) == 0]
    neighbor_dirs = neighbor_dirs.reshape(n_points, n_neighbors, 3)

    # Split this computation into batches due to memory issues.
    batch_size = 2000
    position_uniqueness = np.zeros(n_points)
    direction_uniqueness = np.zeros(n_points)

    for start_idx in torch.arange(0, end=n_points, step=batch_size):
        end_idx = start_idx + batch_size
        
        batch_points_exp = points_expanded[start_idx:end_idx]
        batch_neighbors = neighbors[start_idx:end_idx]

        batch_directions_exp = directions_expanded[start_idx:end_idx]
        batch_neighbor_dirs = neighbor_dirs[start_idx:end_idx]

        # Get vector to centroid of other points; take length as "position
        # uniqueness" score.
        point_to_neighbor = batch_neighbors - batch_points_exp
        point_to_neighbor_centroid = torch.sum(
            point_to_neighbor, axis=1) / n_neighbors
        batch_pos_uniqueness = torch.linalg.norm(
            point_to_neighbor_centroid, axis=1)
        
        # Use same logic to get a score for how different the direction is from the
        # average of other points' directions.
        dir_to_neighbor = batch_neighbor_dirs - batch_directions_exp
        dir_to_neighbor_centroid = torch.sum(
            dir_to_neighbor, axis=1) / n_neighbors
        batch_dir_uniqueness = torch.linalg.norm(
            dir_to_neighbor_centroid, axis=1)

        # Store both in the larger vectors.
        position_uniqueness[start_idx:end_idx] = batch_pos_uniqueness
        direction_uniqueness[start_idx:end_idx] = batch_dir_uniqueness

    return position_uniqueness, direction_uniqueness


def identify_nearest_support_point(support_points: Tensor, sample_points: Tensor
                                   ) -> Tuple[Tensor, Tensor]:
    """Identify the nearest support point for each sample point.

    Args:
        support_points (N, 3):  support points to which samples can snap.
        sample_points (M, 3):  other samples in space, such as those which may
            have been sampled on the surface of an underlying mesh.

    Outputs:
        nearest_support_idx (M):  the index of the nearest support point in
            support_points to the samples in sample_points.
        nearest_support_dists (M):  the distances to the nearest support point.
    """
    n_supports = support_points.shape[0]
    n_samples = sample_points.shape[0]

    # Use 3D arrays indexed by [sample_i, support_i, x/y/z].
    samples_expanded = sample_points.unsqueeze(1).repeat(1, n_supports, 1)
    supports_expanded = support_points.unsqueeze(0).repeat(n_samples, 1, 1)

    # Get (n_samples, n_support, 3) vectors from each sample to a support point.
    sample_to_support_vecs = supports_expanded - samples_expanded

    # Get (n_samples, n_support) distances from each sample to a support point.
    sample_to_support_dists = torch.linalg.norm(sample_to_support_vecs, axis=2)

    # Get (n_samples,) index of support point closest to each sample.
    mins = torch.min(sample_to_support_dists, axis=1)
    nearest_support_idx = mins.indices
    nearest_support_dists = mins.values

    return nearest_support_idx, nearest_support_dists


def rebalance_contact_points(
        mesh: MeshSummary, contact_points: Tensor, contact_directions: Tensor,
        snapping_threshold: float = SUPPORT_POINT_DISTANCE_THRESHOLD
) -> Tuple[Tensor, Tensor]:
    """The contact points may overrepresent certain corners/portions of the
    geometry that struck the ground more often.  This function can help
    rebalance the collection of contact points such that they are more
    geometrically distributed.  This is done by sampling points evenly over the
    area of the underlying mesh surface, then snapping those samples to their
    nearest contact point, rejecting those that have to snap over a distance
    threshold.  This process starts with generating the same number of mesh
    samples as there are contact points, then the rejection sampling reduces the
    total number of points in the rebalanced group.  Retention rates for an
    example experiment (bundlesdf_cube test_004) are:
        - threshold = 10mm:  ~57% retention
        - threshold = 5mm:   ~32% retention
        - threshold = 3mm:   ~18% retention
        - threshold = 1mm:   ~ 4% retention

    Args:
        mesh
        contact_points (N, 3)
        contact_directions (N, 3)
        snapping_threshold

    Outputs:
        balanced_contacts (M, 3) for M <= N
        balanced_directions (M, 3)
    """
    # Sample on the mesh the same number of points as current contacts.
    sample_points, _ = sample_on_mesh(mesh, len(contact_points))

    # Snap these samples to their nearest contact point.
    snapping_idxs, snap_dists = identify_nearest_support_point(
        contact_points, sample_points)
    
    # Rejection sample by removing any points that snapped a distance greater
    # than the threshold.
    good_idxs = snapping_idxs[snap_dists < snapping_threshold]
    balanced_contacts = contact_points[good_idxs]
    balanced_directions = contact_directions[good_idxs]

    return balanced_contacts, balanced_directions


# ============================= Data Generation ============================== #
def generate_point_sdf_pairs(
        points: Tensor, directions: Tensor,
        n_nearby_inside: int = N_QUERY_INSIDE,
        n_nearby_outside: int = N_QUERY_OUTSIDE,
        n_far_outside: int = N_QUERY_OUTSIDE_FAR,
        depth_inside: float = DEPTH_INSIDE,
        depth_outside: float = DEPTH_OUTSIDE,
        depth_far_outside: float = DEPTH_FAR_OUTSIDE
) -> Tuple[Tensor, Tensor]:
    """Generate pairs of 3D points and their associated signed distance, given
    a list of points with known signed distance of zero and directions
    associated with those points' contact directions.

    Args:
        point (N, 3):  N support points of the object geometry for the given
            support directions.
        direction (N, 3):  associated support directions.
        n_nearby_inside (optional):  number of points to query inside geometry
            at a nearby range.
        n_nearby_outside (optional):  number of points to query outside
            geometry at a nearby range.
        n_far_outside (optional):  number of points to query outside geometry at
            a far range.
        depth_inside (optional):  depth in meters for range of sampling interior
            points with n_nearby_inside.
        depth_outside (optional):  depth in meters for range of sampling
            exterior points with n_nearby_outside.
        depth_far_outside (optional):  depth in meters for range of sampling
            exterior points with n_far_outside.

    Outputs:
        points_on_axis (N*M, 3):  M 3D points generated along the ray passing
            through the first provided point in the first provided direction,
            followed by M 3D points w.r.t. the second point and direction, etc.
            Here M = n_nearby_inside + n_nearby_outside + n_far_outside.
        signed_distances (N*M,):  signed distances associated with the points,
            in the same order as points_on_axis.
    """
    # Perform input checks.
    assert points.ndim == directions.ndim == 2, f'Expected 2-dimensional ' \
        f'shapes for {points.shape=} and {directions.shape=}.'
    n_points = points.shape[0]
    assert points.shape == (n_points, 3), f'Expected {points.shape=} to ' \
        + 'be (n_points, 3).'
    assert directions.shape == (n_points, 3), f'Expected {directions.shape=} ' \
        + 'to be (n_points, 3).'
    
    # The signed distances will be tiled such that the first N correspond to
    # point 1, the next N correspond to point 2, etc.
    distance_scalings = torch.cat((
        -depth_inside*torch.ones(n_nearby_inside),
        depth_outside*torch.ones(n_nearby_outside),
        depth_far_outside*torch.ones(n_far_outside)
    )).repeat(n_points)
    signed_distances = distance_scalings * torch.rand_like(distance_scalings)

    n_per_point = n_nearby_inside + n_nearby_outside + n_far_outside

    # Get N repeated for first point, then N repeated for second point, etc.
    repeated_points = points.unsqueeze(1).repeat(1, n_per_point, 1
                                                 ).reshape(-1,3)
    repeated_directions = directions.unsqueeze(1).repeat(1, n_per_point, 1
                                                         ).reshape(-1,3)

    # Get the 3D points corresponding to the generated signed distances.
    points_on_axis = repeated_points + \
        signed_distances.unsqueeze(1).repeat(1,3)*repeated_directions

    return points_on_axis, signed_distances


def generate_point_sdf_bound_pairs(points: Tensor, directions: Tensor
                                   ) -> Tuple[Tensor, Tensor]:
    """Generate pairs of 3D points and their associated minimum signed distance
    bounds, given a list of points with known signed distance of zero and
    directions associated with those points' contact directions.

    Args:
        points (M, 3):  M support points of the object geometry for the given
            support directions.
        directions (M, 3):  associated support directions.

    Outputs:
        points_in_space (M*N, 3):  N 3D points generated randomly, associated
            with the first provided point in the first provided direction,
            followed by N 3D points w.r.t. the second point and direction, etc.
        min_signed_distances (M*N,):  minimum signed distance bounds associated
            with the points, in the same order as points_in_space.
    """
    # Perform input checks.
    assert points.ndim == directions.ndim == 2, f'Expected 2-dimensional ' \
        f'shapes for {points.shape=} and {directions.shape=}.'
    n_points = points.shape[0]
    assert points.shape == (n_points, 3), f'Expected {points.shape=} to ' \
        + 'be (n_points, 3).'
    assert directions.shape == (n_points, 3), f'Expected {directions.shape=} ' \
        + 'to be (n_points, 3).'
    
    # Get one unit vector orthogonal to the provided direction vector.  Can use
    # an intermediate random vector, then cross with `direction` to get an
    # orthogonal one.
    orth_dir_x = torch.nn.functional.normalize(
        torch.cross(directions, torch.rand_like(directions), dim=1), dim=1
    )

    # Get the third unit vector to complete the coordinate system, where the z
    # axis is represented by the input `direction``.
    orth_dir_y = torch.nn.functional.normalize(
        torch.cross(directions, orth_dir_x, dim=1), dim=1
    )

    # Randomly sample radii, heights, and angles.  Everything will be tiled such
    # that the first N correspond to point 1, the next N correspond to point 2,
    # etc.
    n_per_point = BOUNDED_NEARBY_INSIDE_N_QUERY + \
        BOUNDED_NEARBY_OUTSIDE_N_QUERY + BOUNDED_FAR_N_QUERY
    radius_scalings = torch.cat((
        BOUNDED_NEARBY_RADIUS*torch.ones(BOUNDED_NEARBY_INSIDE_N_QUERY),
        BOUNDED_NEARBY_RADIUS*torch.ones(BOUNDED_NEARBY_OUTSIDE_N_QUERY),
        BOUNDED_FAR_RADIUS*torch.ones(BOUNDED_FAR_N_QUERY)
    )).repeat(n_points)
    height_scalings = torch.cat((
        -BOUNDED_NEARBY_DEPTH*torch.ones(BOUNDED_NEARBY_INSIDE_N_QUERY),
        BOUNDED_NEARBY_DEPTH*torch.ones(BOUNDED_NEARBY_OUTSIDE_N_QUERY),
        BOUNDED_FAR_DEPTH*torch.ones(BOUNDED_FAR_N_QUERY)
    )).repeat(n_points)
    angle_scalings = 2*torch.pi*torch.ones(n_points).repeat(n_per_point)
    radii = radius_scalings * torch.rand_like(radius_scalings)
    heights = height_scalings * torch.rand_like(height_scalings)
    angles = angle_scalings * torch.rand_like(angle_scalings)

    repeated_points = points.unsqueeze(1).repeat(1, n_per_point, 1
                                                ).reshape(-1, 3)
    repeated_zs = directions.unsqueeze(1).repeat(1, n_per_point, 1
                                                ).reshape(-1, 3)
    repeated_xs = orth_dir_x.unsqueeze(1).repeat(1, n_per_point, 1
                                                ).reshape(-1, 3)
    repeated_ys = orth_dir_y.unsqueeze(1).repeat(1, n_per_point, 1
                                                ).reshape(-1, 3)
    repeated_radii = radii.unsqueeze(1).repeat(1,3)
    repeated_heights = heights.unsqueeze(1).repeat(1,3)
    repeated_angles = angles.unsqueeze(1).repeat(1,3)

    # Construct the 3D points from the radii, heights, and angles.
    points_in_space = repeated_points + \
        repeated_heights*repeated_zs + \
        repeated_radii*torch.cos(repeated_angles)*repeated_xs + \
        repeated_radii*torch.sin(repeated_angles)*repeated_ys

    min_signed_distances = heights

    return points_in_space, min_signed_distances


def generate_training_data(points: Tensor, directions: Tensor) -> None:
    """Given points and directions, create points with SDF values or bounds for
    training BundleSDF.
    
    Args:
        points (N, 3):  observed support points from ContactNets.
        directions (N, 3):  directions associated with the provided points.

    Outputs:
        ps (M, 3):  new points generated along rays through support points along
            the provided directions.
        sdfs (M,):  signed distances associated with the generated ps.
        vs (L, 3):  new points generated randomly in the neighborhood around
            support points.
        sdf_bounds (L,):  minimum signed distance bounds associated with the
            generated vs.
    """
    # Do some input checking.
    assert points.shape == directions.shape
    assert points.ndim == directions.ndim == 2
    assert points.shape[1] == directions.shape[1] == 3

    # Compute the outputs.
    ps, sdfs = generate_point_sdf_pairs(points, directions)
    vs, sdf_bounds = generate_point_sdf_bound_pairs(points, directions)

    return ps, sdfs, vs, sdf_bounds


def generate_point_sdf_gradient_pairs(points: Tensor, normals: Tensor
                                      ) -> Tuple[Tensor, Tensor]:
    """Given a set of points and their associated outward normals, generate an
    sampled set of points and their associated outward normals.  These (point,
    normal) pairs are to be used in BundleSDF loss that enforces the SDF should
    monotonically increase along a normal direction outside (and a little bit
    inside) a geometry's convex hull.

    Args:
        points (N, 3):  points on the surface of the geometry's convex hull.
        normals (N, 3):  outward normals associated with the points.

    Outputs:
        extended_points (N*M, 3):  a set of points obtained by walking along the
            normal direction associated with each point.  Note that M =
            n_nearby_inside + n_nearby_outside + n_far_outside values passed
            into generate_point_sdf_pairs.
        extended_normals (N*M, 3):  the normals associated with the
            extended_points.
    """
    # Can generate points in the same way as generating the (point, SDF) pairs.
    extended_points, _ = generate_point_sdf_pairs(
        points, normals,
        n_nearby_inside=GRADIENT_N_QUERY_INSIDE,
        n_nearby_outside=GRADIENT_N_QUERY_OUTSIDE,
        n_far_outside=GRADIENT_N_QUERY_OUTSIDE_FAR,
        depth_inside=GRADIENT_DEPTH_INSIDE,
        depth_outside=GRADIENT_DEPTH_OUTSIDE,
        depth_far_outside=GRADIENT_DEPTH_FAR_OUTSIDE
    )

    # Repeat the normals according to the tesselation expected from the above
    # output (which is the first M are for point 1, etc.).
    n_points = points.shape[0]
    n_samples_per_point = GRADIENT_N_QUERY_INSIDE + GRADIENT_N_QUERY_OUTSIDE + \
        GRADIENT_N_QUERY_OUTSIDE_FAR
    extended_normals = normals.repeat_interleave(n_samples_per_point, dim=0)

    # Sanity check that these shapes match.
    assert extended_points.shape == extended_normals.shape == \
        (n_samples_per_point*n_points, 3)
    
    return extended_points, extended_normals


def generate_training_data_for_run(run_name: str, storage_name: str):
    # Load the exported outputs from the experiment run.
    output_dir = file_utils.geom_for_bsdf_dir(storage_name, run_name)
    normal_forces = torch.load(
        op.join(output_dir, EXPORT_FORCES_DEFAULT_NAME)).detach()
    support_points = torch.load(
        op.join(output_dir, EXPORT_POINTS_DEFAULT_NAME)).detach()
    support_directions = torch.load(
        op.join(output_dir, EXPORT_DIRECTIONS_DEFAULT_NAME)).detach()

    # Sample points on the support point mesh surface and visualize them.
    mesh = create_mesh_from_set_of_points(support_points)
    sample_points, sample_normals = sample_on_mesh(mesh, N_MESH_SAMPLE)

    # Filter support points via simple thresholding of normal forces, then
    # filter the sample points based on this contact knowledge.
    contact_points, contact_directions = \
        filter_points_and_directions_based_on_contact(
            support_points, support_directions, normal_forces
        )
    sample_points_cf, sample_normals_cf = filter_mesh_samples_based_on_supports(
        sample_points, sample_normals, contact_points, contact_directions
    )

    # Generate training data for the mesh sample points.
    mesh_ps, mesh_sdfs = generate_point_sdf_pairs(
        sample_points_cf, sample_normals_cf,
        n_nearby_inside=MESH_N_QUERY_INSIDE,
        n_nearby_outside=MESH_N_QUERY_OUTSIDE,
        n_far_outside=MESH_N_QUERY_OUTSIDE_FAR,
        depth_inside=MESH_DEPTH_INSIDE,
        depth_outside=MESH_DEPTH_OUTSIDE,
        depth_far_outside=MESH_DEPTH_FAR_OUTSIDE
    )
    mesh_vs, mesh_sdf_bounds = generate_point_sdf_bound_pairs(
        sample_points_cf, sample_normals_cf
    )

    # Generate SDF gradient training data from the mesh sample points.
    mesh_ws, mesh_w_normals = generate_point_sdf_gradient_pairs(
        sample_points_cf, sample_normals_cf
    )

    # Redistribute the contact points to be more geometrically balanced (this
    # will decrease the number of contacts/directions considered).
    balanced_contacts, balanced_directions = rebalance_contact_points(
        mesh, contact_points, contact_directions
    )
    
    # Generate training data from the redistributed contact points.
    contact_ps, contact_sdfs, contact_vs, contact_sdf_bounds = \
        generate_training_data(balanced_contacts, balanced_directions)
    
    # Save the generated data.
    file_utils.store_sdf_for_bsdf(
        storage_name, run_name,
        from_support_not_mesh=False,
        ps=mesh_ps, sdfs=mesh_sdfs,
        vs=mesh_vs, sdf_bounds=mesh_sdf_bounds,
        ws=mesh_ws, w_normals=mesh_w_normals
    )
    file_utils.store_sdf_for_bsdf(
        storage_name, run_name,
        from_support_not_mesh=True,
        ps=contact_ps, sdfs=contact_sdfs,
        vs=contact_vs, sdf_bounds=contact_sdf_bounds
    )


# ============================== Visualization =============================== #
def visualize(ps,sdfs):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    colored_sdfs = ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], c=sdfs,
                              cmap='viridis', marker='o',
                              label='Points with assigned SDF')
    cbar = fig.colorbar(colored_sdfs)
    cbar.set_label('SDF')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.legend()
    plt.show()


def visualize_sdfs(points: Tensor, directions: Tensor, ps: Tensor = None,
                   sdfs: Tensor = None, vs: Tensor = None,
                   sdf_bounds: Tensor = None) -> None:
    """Visualize the generated data from provided points and associated
    directions.  If the data is not already generated, this function will do so.

    Args:
        points (N, 3):  observed support points from ContactNets.
        directions (N, 3):  directions associated with the provided points.
        ps (M, 3):  new points generated along rays through support points along
            the provided directions.
        sdfs (M,):  signed distances associated with the generated ps.
        vs (L, 3):  new points generated randomly in the neighborhood around
            support points.
        sdf_bounds (L,):  minimum signed distance bounds associated with the
            generated vs.
    """
    # Check if any of the inputs weren't provided and need to be generated.
    if ps is None or sdfs is None:
        # Generate SDF points along axis.
        ps, sdfs = generate_point_sdf_pairs(points, directions)
    if vs is None or sdf_bounds is None:
        # Generate more bounded SDFs.
        vs, sdf_bounds = generate_point_sdf_bound_pairs(points, directions)

    # Do some input checking.
    assert points.shape == directions.shape
    assert ps.shape[0] == sdfs.shape[0]
    assert vs.shape[0] == sdf_bounds.shape[0]
    assert points.ndim == ps.ndim == vs.ndim == 2
    assert sdfs.ndim == sdf_bounds.ndim == 1
    assert points.shape[1] == ps.shape[1] == vs.shape[1] == 3

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the original support point and direction.
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='*', s=20,
               color='r', label='Support points')
    prefix = [''] + ['_']*(len(directions)-1)
    for i in range(len(directions)):
        ax.quiver(*points[i], *directions[i]/4, color='r',
                  label=prefix[i]+'Support directions')

    # Plot the generated data and their associated SDFs.
    colored_sdfs = ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], c=sdfs,
                              cmap='viridis', marker='o',
                              label='Points with assigned SDF')
    # ax.scatter(vs[:, 0], vs[:, 1], vs[:, 2], c=sdf_bounds, cmap='viridis',
    #            marker='.', label='Points with SDF bound')

    # Because both scatter series are using the 'viridis' color map, the
    # colorbar will share a mapping for both series.
    cbar = fig.colorbar(colored_sdfs)
    cbar.set_label('SDF')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.legend()

    # Set equal aspect ratio.
    ax.set_box_aspect([np.ptp(arr) for arr in \
                       [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])
    plt.show()


def visualize_sampled_points(
        mesh: MeshSummary, sampled_points: Tensor, sampled_normals: Tensor,
        filtered_points: Tensor = None, filtered_normals: Tensor = None,
        support_points: Tensor = None, support_directions: Tensor = None
) -> None:
    """Visualize the sample points generated on the provided mesh.  Can help
    visually verify the points are strictly on the mesh surface with normals
    facing outwards.  Optionally can show how the points are filtered based on
    support points.

    Args:
        mesh:  a MeshSummary of the object geometry.  Won't visualize the mesh
            surfaces if mesh is None (which can be beneficial for large meshes).
        sampled_points (N, 3):  points sampled on the mesh surface.
        sampled_normals (N, 3):  outward normals associated with the points.
        filtered_points (M, 3):  (optional) a subset of sampled_points
        filtered_normals (M, 3):  (optional) a subset of sampled_normals
        support_points (K, 3):  (optional) support points.
        support_directions (K, 3):  (optional) support directions.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the mesh wireframe.
    if mesh is not None:
        prefix = [''] + ['_']*(len(mesh.faces)-1)
        for i in range(len(mesh.faces)):
            face = mesh.faces[i]
            vertices = mesh.vertices[face]
            vertices = torch.cat((vertices, vertices[0].unsqueeze(0)), dim=0)
            vertices = vertices.numpy()
            ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='b',
                    label=prefix[i]+'Mesh edges')

    # Plot the sampled points and outward normals.
    color = '#00000044' if filtered_points is not None else 'r'
    ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2],
               marker='*', s=20, color=color, label='Sample points')
    prefix = [''] + ['_']*(len(sampled_normals)-1)
    for i in range(len(sampled_normals)):
        ax.quiver(*sampled_points[i], *sampled_normals[i]/25, color=color,
                  label=prefix[i]+'Outward normals', zorder=1.5)
        
    if filtered_points is not None:
        ax.scatter(filtered_points[:, 0], filtered_points[:, 1],
                   filtered_points[:, 2], marker='*', s=30, color='r',
                   label='Filtered samples')
        prefix = [''] + ['_']*(len(filtered_normals)-1)
        for i in range(len(filtered_normals)):
            ax.quiver(*filtered_points[i], *filtered_normals[i]/20, color='r',
                    label=prefix[i]+'Outward filter normals', zorder=1.5)
            
    if support_points is not None:
        ax.scatter(support_points[:, 0], support_points[:, 1],
                   support_points[:, 2], marker='*', s=40, color='g',
                   label='Support points')
        prefix = [''] + ['_']*(len(support_directions)-1)
        for i in range(len(support_directions)):
            ax.quiver(*support_points[i], *support_directions[i]/10,
                      color='g', label=prefix[i]+'Support directions',
                      zorder=1.5)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.legend()

    # Set equal aspect ratio.
    ax.set_box_aspect([np.ptp(arr) for arr in \
                       [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])
    plt.show()


def visualize_gradients(mesh: MeshSummary, sample_points: Tensor,
                        points_with_grad: Tensor, point_grads: Tensor) -> None:
    """Visualize points generated for SDF gradient supervision.  Can show the
    underlying PLL mesh, points sampled on the mesh, and sampled points along
    the mesh normal with shared normal vectors, i.e. SDF gradients.
    
    Args:
        mesh:  PLL's object geometry as a mesh.  This can be None and the mesh
            wireframe will not be drawn.
        sample_points (N, 3):  points sampled on the surface of the mesh.
        points_with_grad (M*N, 3):  points sampled along axes passing through
            the sample_points in the mesh normal direction.
        point_grads (M*N, 3):  mesh normal directions i.e. SDF gradients at each
            of the points_with_grad.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the mesh wireframe.
    if mesh is not None:
        prefix = [''] + ['_']*(len(mesh.faces)-1)
        for i in range(len(mesh.faces)):
            face = mesh.faces[i]
            vertices = mesh.vertices[face]
            vertices = torch.cat((vertices, vertices[0].unsqueeze(0)), dim=0)
            vertices = vertices.numpy()
            ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='b',
                    label=prefix[i]+'Mesh edges')

    # Plot the sampled points and outward normals.
    ax.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2],
               marker='*', s=20, color='r', label='Mesh sample points')
    prefix = [''] + ['_']*(len(points_with_grad)-1)
    for i in range(len(point_grads)):
        ax.quiver(*points_with_grad[i], *point_grads[i]/25, color='g',
                  label=prefix[i]+'SDF gradient', zorder=1.5)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.legend()

    # Set equal aspect ratio.
    ax.set_box_aspect([np.ptp(arr) for arr in \
                       [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])
    plt.show()


def visualize_point_uniquenesses(
        points: Tensor, directions: Tensor, position_uniqueness: Tensor,
        direction_uniqueness: Tensor, second_points: Tensor = None,
        second_directions: Tensor = None, second_pos_uniqueness: Tensor = None,
        second_dir_uniqueness: Tensor = None) -> None:
    """Visualize a colorized mapping of point locations/directions and their
    position/direction uniqueness scores.  If provided two sets of everything,
    will visualize both 3D results side-by-side with the same colorbar scale.

    Args:
        points (N, 3)
        directions (N, 3)
        position_uniqueness (N,)
        direction_uniqueness (N,)
        second_points (M, 3)
        second_directions (M, 3)
        second_pos_uniqueness (M,)
        second_dir_uniqueness (M,)
    """
    # See if make 1 or 2 side-by-side plots.
    point_groups = [points]
    direction_groups = [directions]
    position_uniqueness_groups = [position_uniqueness]
    direction_uniqueness_groups = [direction_uniqueness]
    subtitles = [None]
    n_plots = 1
    if second_points is not None:
        point_groups.append(second_points)
        direction_groups.append(second_directions)
        position_uniqueness_groups.append(second_pos_uniqueness)
        direction_uniqueness_groups.append(second_dir_uniqueness)
        subtitles = ['Original', 'Redistributed']
        n_plots = 2

    def do_uniqueness_subplot(
            ax: Axes3D, pts: Tensor, dirs: Tensor, uniqueness: Tensor,
            is_position_not_direction: bool) -> Path3DCollection:
        """Helper function to add a 3D uniqueness plot to a given axis.  Need to
        flag if position or direction uniqueness via is_position_not_direction.
        Returns the Path3DCollection object which can be used later for creating
        a color bar."""
        # Do some input checking.
        assert pts.shape == dirs.shape
        assert pts.shape[0] == uniqueness.shape[0]
        assert pts.ndim == 2
        assert uniqueness.ndim == 1
        assert pts.shape[1] == 3
        assert type(ax) == Axes3D

        if is_position_not_direction:
            color_scale = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], 
                                     c=uniqueness, cmap='rainbow', marker='o')
            
        else:
            color_scale = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                                     c=uniqueness, cmap='rainbow', marker='o')
            for i in range(len(dirs)):
                color = color_scale.to_rgba(uniqueness[i])
                ax.quiver(*pts[i], *dirs[i]/100, color=color, zorder=1.5)

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        # Set equal aspect ratio.
        ax.set_box_aspect([np.ptp(arr) for arr in \
                        [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])
        
        return color_scale

    # Do position uniqueness plot first.
    zipped = zip(point_groups, direction_groups, position_uniqueness_groups,
                 subtitles)
    
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2, ncols=n_plots, height_ratios=[1, 0.05])
    for i, (pts, dirs, pos_uniq, subtitle) in enumerate(zipped):
        ax = fig.add_subplot(gs[0, i], projection='3d')
        color_scale = do_uniqueness_subplot(ax, pts, dirs, pos_uniq, True)
        ax.set_title(subtitle)

    cbar = fig.colorbar(color_scale, cax=fig.add_subplot(gs[1, :]),
                        orientation='horizontal')
    cbar.set_label('Position uniqueness')
    plt.show()

    # Do direction uniqueness plot next.
    # Surprisingly, reusing the old zip doesn't work.
    zipped = zip(point_groups, direction_groups,
                 direction_uniqueness_groups, subtitles)
    
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2, ncols=n_plots, height_ratios=[1, 0.05])
    for i, (pts, dirs, dir_uniq, subtitle) in enumerate(zipped):
        ax = fig.add_subplot(gs[0, i], projection='3d')
        color_scale = do_uniqueness_subplot(ax, pts, dirs, dir_uniq, False)
        ax.set_title(subtitle)

    cbar = fig.colorbar(color_scale, cax=fig.add_subplot(gs[1, :]),
                        orientation='horizontal')
    cbar.set_label('Direction uniqueness')
    plt.show()


def visualize_point_snapping(
        mesh: MeshSummary, support_points: Tensor, sample_points: Tensor,
        snapping_idxs: Tensor, sampling_dists: Tensor,
        dist_threshold: float) -> None:
    """Visualize the result of snapping points sampled on a mesh surface to
    their nearest support points.

    Args:
        mesh:  can be None then wireframe will not get drawn.
        support_points (N, 3)
        sample_points (M, 3)
        snapping_idxs (M,)
        sampling_dists (M,)
        dist_threshold:  if sampling_dists is above the threshold, then plot
            the snapping in gray to indicate those will be rejected.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the mesh wireframe.
    if mesh is not None:
        prefix = [''] + ['_']*(len(mesh.faces)-1)
        for i in range(len(mesh.faces)):
            face = mesh.faces[i]
            vertices = mesh.vertices[face]
            vertices = torch.cat((vertices, vertices[0].unsqueeze(0)), dim=0)
            vertices = vertices.numpy()
            ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    color='b', linewidth=0.1, label=prefix[i]+'Mesh edges')
            
    # Plot the support points.
    ax.scatter(support_points[:, 0], support_points[:, 1], support_points[:, 2], 
               color='b', marker='o', label='Support points')
    
    # Plot the sample points.
    ax.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], 
               color='r', marker='+', label='Sample points')
    
    # Plot the snapping of each sample point to nearest support point.
    reject_labeled, accept_labeled = False, False
    labels = ['Rejected snap travel', 'Retained snap travel']
    for i in range(len(sample_points)):
        snap_dist = sampling_dists[i]
        accept = snap_dist < dist_threshold
        color = 'cyan' if accept else 'gray'
        linewidth = 2 if accept else 0.5
        zorder = 1.8 if accept else 1.5
        prefix = '_' if ((reject_labeled and not accept) or \
                         (accept_labeled and accept)) else ''
        label = prefix + labels[accept]

        if accept:
            accept_labeled = True
        else:
            reject_labeled = True

        support_point = support_points[snapping_idxs[i]]

        xs = [sample_points[i][0], support_point[0]]
        ys = [sample_points[i][1], support_point[1]]
        zs = [sample_points[i][2], support_point[2]]
        ax.plot(xs, ys, zs, color=color, linewidth=linewidth, label=label,
                zorder=zorder)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.legend()

    # Set equal aspect ratio.
    ax.set_box_aspect([np.ptp(arr) for arr in \
                       [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])
    plt.show()


# ============================ Loading Management ============================ #
def load_run_data(run_name: str, system: str) -> None:
    storage_name = file_utils.assure_created(
        op.join(file_utils.RESULTS_DIR, system))

    # Load the exported outputs from the experiment run.
    output_dir = file_utils.geom_for_bsdf_dir(storage_name, run_name)
    normal_forces = torch.load(
        op.join(output_dir, EXPORT_FORCES_DEFAULT_NAME)).detach()
    support_points = torch.load(
        op.join(output_dir, EXPORT_POINTS_DEFAULT_NAME)).detach()
    support_directions = torch.load(
        op.join(output_dir, EXPORT_DIRECTIONS_DEFAULT_NAME)).detach()
    
    return support_points, support_directions, normal_forces, output_dir


def load_deep_support_convex_network(run_name: str, system: str
                                     ) -> HomogeneousICNN:
    """Load a deep support convex network stored from a previous experiment run,
    corresponding to that run's best learned system state.

    Args:
        run_name:  name of the run whose results are to be loaded.
        system:  name of the system.

    Outputs:
        A deep support convex network in the form of a HomogemeousICNN.
    """
    storage_name = file_utils.assure_created(
        op.join(file_utils.RESULTS_DIR, system))
    
    checkpoint_filename = file_utils.get_model_filename(storage_name, run_name)
    checkpoint_dict = torch.load(checkpoint_filename)

    geom_model_dict = {}
    for key, val in checkpoint_dict['best_learned_system_state'].items():
        if 'geometries.0.network.' in key:
            geom_model_dict[key.split('geometries.0.network.')[-1]] = val

    deep_support = HomogeneousICNN(depth = _DEEP_SUPPORT_DEFAULT_DEPTH,
                                   width = _DEEP_SUPPORT_DEFAULT_WIDTH)
    deep_support.load_state_dict(geom_model_dict)

    return deep_support


# ======================== End of Function Definitions ======================= #


# Tests.
if DO_SMALL_FILTERING_AND_VISUALIZATION_TEST:
    print('Performing small filtering and visualization test.')
    point_set = Tensor([[0.6, 0, 0], [0.6, 1, 0], [0, 1, 0], [0, 0, 0],
                        [0.6, 0, 1], [0.6, 1, 1], [0, 1, 1], [0, 0, 1]])
    mesh = create_mesh_from_set_of_points(point_set)

    sample_points, sample_normals = sample_on_mesh(mesh, 50)
    print(f'\tVisualizing points sampled on mesh.')
    visualize_sampled_points(mesh, sample_points, sample_normals)

    support_points = Tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    support_dirs = Tensor([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    sample_points_cf, sample_normals_cf = filter_mesh_samples_based_on_supports(
        sample_points, sample_normals, support_points, support_dirs
    )
    print('\tVisualizing points sampled with filtering via support points.')
    visualize_sampled_points(
        mesh, sample_points, sample_normals, filtered_points=sample_points_cf,
        filtered_normals=sample_normals_cf, support_points=support_points,
        support_directions=support_dirs
    )
    print('\tDeleting test variables so can\'t accidentally be reused.')
    del point_set, mesh, sample_points, sample_normals, support_points, \
        support_dirs, sample_points_cf, sample_normals_cf
    print('Done with small filtering and visualization test.')
    
if DO_SDFS_FROM_MESH_SAMPLING_WITH_SUPPORT_FILTERING:
    print('Performing SDF generation from mesh sampling with contact ' + \
          'filtering test.')
    point_set = Tensor([[0.6, 0, 0], [0.6, 1, 0], [0, 1, 0], [0, 0, 0],
                        [0.6, 0, 1], [0.6, 1, 1], [0, 1, 1], [0, 0, 1]])
    mesh = create_mesh_from_set_of_points(point_set)

    sample_points, sample_normals = sample_on_mesh(mesh, 50)
    print(f'\tVisualizing points sampled on mesh.')
    visualize_sampled_points(mesh, sample_points, sample_normals)

    support_points = Tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    support_dirs = Tensor([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    sample_points_cf, sample_normals_cf = filter_mesh_samples_based_on_supports(
        sample_points, sample_normals, support_points, support_dirs
    )
    print('\tVisualizing points sampled with filtering via support points.')
    visualize_sampled_points(
        mesh, sample_points, sample_normals, filtered_points=sample_points_cf,
        filtered_normals=sample_normals_cf, support_points=support_points,
        support_directions=support_dirs
    )

    print('\tGenerating (point, SDF) pairs from mesh samples.')
    ps, sdfs = generate_point_sdf_pairs(
        sample_points_cf, sample_normals_cf,
        n_nearby_inside=MESH_N_QUERY_INSIDE,
        n_nearby_outside=MESH_N_QUERY_OUTSIDE,
        n_far_outside=MESH_N_QUERY_OUTSIDE_FAR,
        depth_inside=MESH_DEPTH_INSIDE,
        depth_outside=MESH_DEPTH_OUTSIDE,
        depth_far_outside=MESH_DEPTH_FAR_OUTSIDE
    )

    print('\tVisualizing the samples.')
    visualize_sdfs(sample_points_cf, sample_normals_cf, ps=ps, sdfs=sdfs)

    print('\tDeleting test variables so can\'t accidentally be reused.')
    del point_set, mesh, sample_points, sample_normals, support_points, \
        support_dirs, sample_points_cf, sample_normals_cf, ps, sdfs
    print('Done with SDF generation from mesh and contact filtering test.')

if DO_COMBINE_SUPPORT_POINTS_AND_MESH_SAMPLING:
    print('Performing combining support points and mesh samples test.')

    print(f'\tLoading results from {TEST_RUN_NAME} in {SYSTEM_NAME}.')
    support_points, support_directions, normal_forces, output_dir = \
        load_run_data(TEST_RUN_NAME, SYSTEM_NAME)
    
    # Sample points on the mesh surface and visualize them.
    print('\tSampling points on the mesh surface.')
    mesh = create_mesh_from_set_of_points(support_points)
    sample_points, sample_normals = sample_on_mesh(mesh, 100)
    
    # Perform filtering via simple thresholding of normal forces.
    print('\tFiltering based on inferred contact.')
    contact_points, contact_directions = filter_points_and_directions_based_on_contact(
        support_points, support_directions, normal_forces)

    # Generate training data.
    mesh_ps, mesh_sdfs, mesh_vs, mesh_sdf_bounds = \
        generate_training_data(sample_points, sample_normals)
    contact_ps, contact_sdfs, contact_vs, contact_sdf_bounds = \
        generate_training_data(contact_points, contact_directions)
    ps = torch.cat((mesh_ps, contact_ps), dim=0)
    sdfs = torch.cat((mesh_sdfs, contact_sdfs), dim=0)
    vs = torch.cat((mesh_vs, contact_vs), dim=0)
    sdf_bounds = torch.cat((mesh_sdf_bounds, contact_sdf_bounds), dim=0)

    print(f'\tGenerated training data: \n\t\t{mesh_ps.shape=}')
    print(f'\t\t{mesh_vs.shape=} \n\t\t{contact_ps.shape=}')
    print(f'\t\t{contact_vs.shape=}')

    # Visualize it.  Note:  can call this visualization function without
    # providing the training data, and it will generate some for visualization
    # purposes.
    visualize_sdfs(contact_points, contact_directions, ps=ps, sdfs=sdfs, vs=vs,
                   sdf_bounds=sdf_bounds)

    print('\tDeleting test variables so can\'t accidentally be reused.')
    del support_points, support_directions, mesh, sample_points, \
        sample_normals, contact_points, contact_directions, mesh_ps, \
        mesh_sdfs, mesh_vs, mesh_sdf_bounds, contact_ps, contact_sdfs, \
        contact_vs, contact_sdf_bounds, ps, sdfs, vs, sdf_bounds
    print('Done with combining support points and mesh samples test.')

if DO_NETWORK_LOADING_TEST:
    print('Performing loading deep support convex network test.')

    # Can load a pre-trained deep support convex network.
    network = load_deep_support_convex_network(TEST_RUN_NAME, SYSTEM_NAME)
    mesh = create_mesh_from_deep_support(network)

    print('\tDeleting test variables so can\'t accidentally be reused.')
    del network, mesh
    print('Done with loading deep support convex network test.')

if DO_GRADIENT_DATA_TEST:
    print('Performing gradient data generation test.')
    support_points, support_directions, normal_forces, _ = \
        load_run_data(TEST_RUN_NAME, SYSTEM_NAME)

    # Sample points on the support point mesh surface and visualize them.
    mesh = create_mesh_from_set_of_points(support_points)
    sample_points, sample_normals = sample_on_mesh(mesh, 100)

    # Filter support points via simple thresholding of normal forces, then
    # filter the sample points based on this contact knowledge.
    contact_points, contact_directions = filter_points_and_directions_based_on_contact(
        support_points, support_directions, normal_forces)
    sample_points_cf, sample_normals_cf = filter_mesh_samples_based_on_supports(
        sample_points, sample_normals, contact_points, contact_directions
    )

    # Generate SDF gradient training data from the mesh sample points.
    mesh_ws, mesh_w_normals = generate_point_sdf_gradient_pairs(
        sample_points_cf, sample_normals_cf
    )

    # Visualize.
    visualize_gradients(mesh, sample_points_cf, mesh_ws, mesh_w_normals)

    print('\tDeleting test variables so can\'t accidentally be reused.')
    del support_points, support_directions, normal_forces, mesh, \
        sample_points, sample_normals, contact_points, contact_directions, \
        sample_points_cf, sample_normals_cf, mesh_ws, mesh_w_normals
    print('Done with gradient data generation test.')

if DO_UNIQUENESS_SCORE_TEST:
    print('Performing uniqueness score test.')
    support_points, support_directions, _, _ = \
        load_run_data(TEST_RUN_NAME, SYSTEM_NAME)
    
    print('\tComputing uniqueness scores.')
    pos_uniq, dir_uniq = compute_position_and_direction_uniquenesses(
        support_points, support_directions
    )

    print('\tVisualizing position then direction uniqueness.')
    visualize_point_uniquenesses(support_points, support_directions, pos_uniq,
                                 dir_uniq)

    print('\tDeleting test variables so can\'t accidentally be reused.')
    del support_points, support_directions, pos_uniq, dir_uniq
    print('Done with uniqueness score test.')

if DO_DOUBLE_UNIQUENESS_SCORE_TEST:
    print('Performing double uniqueness score test.')
    support_points, support_directions, _, _ = \
        load_run_data(TEST_RUN_NAME, SYSTEM_NAME)
    
    print('\tComputing uniqueness scores.')
    pos_uniq, dir_uniq = compute_position_and_direction_uniquenesses(
        support_points, support_directions
    )

    print('\tVisualizing position then direction uniqueness.')
    visualize_point_uniquenesses(
        support_points, support_directions, pos_uniq, dir_uniq,
        second_points=support_points, second_directions=support_directions,
        second_pos_uniqueness=pos_uniq, second_dir_uniqueness=dir_uniq)

    print('\tDeleting test variables so can\'t accidentally be reused.')
    del support_points, support_directions, pos_uniq, dir_uniq
    print('Done with double uniqueness score test.')

if DO_SUPPORT_POINT_SNAPPING_TEST:
    print('Performing support point snapping test.')

    # Load the support points and mesh from a finished run.
    print(f'\tLoading support points and generating mesh from support network.')
    support_points, support_directions, _, _ = \
        load_run_data(TEST_RUN_NAME, SYSTEM_NAME)
    network = load_deep_support_convex_network(TEST_RUN_NAME, SYSTEM_NAME)
    mesh = create_mesh_from_deep_support(network)

    print(f'\tSampling on mesh and snapping to nearest support points.')
    samples, _ = sample_on_mesh(mesh, n_sample=len(support_points))
    snapping_idxs, snap_dists = identify_nearest_support_point(
        support_points, samples)
    
    print(f'\tReject samples that need to travel too far to snap.')
    good_idxs = snapping_idxs[snap_dists < SUPPORT_POINT_DISTANCE_THRESHOLD]
    print(f'\t-> Went from {len(snapping_idxs)} to {len(good_idxs)} ' + \
          f'with snap distance of {SUPPORT_POINT_DISTANCE_THRESHOLD}m;\n' + \
          f'\t   {len(good_idxs)/len(snapping_idxs)*100:.2f}% retention rate.')

    print(f'\tVisualizing the generated points and snapped supports.')
    visualize_point_snapping(
        mesh, support_points, samples, snapping_idxs, snap_dists,
        dist_threshold=SUPPORT_POINT_DISTANCE_THRESHOLD)

    print(f'\tComputing uniqueness values of original supports.')
    orig_pos_uniq, orig_dir_uniq = compute_position_and_direction_uniquenesses(
        support_points, support_directions
    )
    print(f'\tComputing uniqueness values of snapped samples.')
    snap_points = support_points[good_idxs]
    snap_directions = support_directions[good_idxs]
    snap_pos_uniq, snap_dir_uniq = compute_position_and_direction_uniquenesses(
        snap_points, snap_directions)

    print(f'\tVisualizing uniqueness before and after geometry-based sampling.')
    visualize_point_uniquenesses(
        support_points, support_directions, orig_pos_uniq, orig_dir_uniq,
        second_points=snap_points, second_directions=snap_directions,
        second_pos_uniqueness=snap_pos_uniq,
        second_dir_uniqueness=snap_dir_uniq)

    print('\tDeleting test variables so can\'t accidentally be reused.')
    del support_points, support_directions, network, mesh, samples, \
        snapping_idxs, orig_pos_uniq, orig_dir_uniq, snap_points, \
        snap_directions, snap_pos_uniq, snap_dir_uniq
    print('Done with support point snapping test.')


if __name__ == '__main__':
    pdb.set_trace()

    # Generate training data for run.
    storage_name = file_utils.assure_created(
        op.join(file_utils.RESULTS_DIR, SYSTEM_NAME))
    generate_training_data_for_run(TEST_RUN_NAME, storage_name)
