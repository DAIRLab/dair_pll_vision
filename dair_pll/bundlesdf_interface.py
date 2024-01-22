"""Functionality related to connecting ContactNets to BundleSDF."""

import os
import os.path as op
import pdb
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from dair_pll.system import MeshSummary
import torch
from torch import Tensor
from scipy.spatial import ConvexHull  # type: ignore

from dair_pll import deep_support_function, file_utils
from dair_pll.geometry import _DEEP_SUPPORT_DEFAULT_DEPTH, \
    _DEEP_SUPPORT_DEFAULT_WIDTH
from dair_pll.deep_support_function import HomogeneousICNN


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

# Hyperparameters for querying around an object with SDF minimum bounds.
BOUNDED_NEARBY_DEPTH = 0.005
BOUNDED_NEARBY_RADIUS = 0.05
BOUNDED_FAR_DEPTH = 0.1
BOUNDED_FAR_RADIUS = 0.1
BOUNDED_FAR_N_QUERY = 100
BOUNDED_NEARBY_OUTSIDE_N_QUERY = 100
# Make the inside queried points equal to all queried outside.
BOUNDED_NEARBY_INSIDE_N_QUERY = BOUNDED_FAR_N_QUERY + BOUNDED_NEARBY_OUTSIDE_N_QUERY

# Hyperparameters for filtering support points or hull sample points
FORCE_THRESH = 0.3676               # Newtons
HULL_PROXIMITY_THRESH = 0.001       # meters

# Flags for running some unit tests.
DO_SMALL_FILTERING_AND_VISUALIZATION_TEST = False
DO_SDFS_FROM_MESH_SAMPLING_WITH_CONTACT_FILTERING = True


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
        point (M, 3):  M support points of the object geometry for the given
            support directions.
        direction (M, 3):  associated support directions.
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
        points_on_axis (M*N, 3):  N 3D points generated along the ray passing
            through the first provided point in the first provided direction,
            followed by N 3D points w.r.t. the second point and direction, etc.
        signed_distances (M*N,):  signed distances associated with the points,
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


def filter_pts_and_dirs(
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


def load_deep_support_convex_network(storage_name: str, run_name: str
                                     ) -> HomogeneousICNN:
    """Load a deep support convex network stored from a previous experiment run,
    corresponding to that run's best learned system state.

    Args:
        storage_name:  name of the storage directory.
        run_name:  name of the run whose results are to be loaded.

    Outputs:
        A deep support convex network in the form of a HomogemeousICNN.
    """
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
        mesh:  a MeshSummary of the object geometry.
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
    prefix = [''] + ['_']*(len(mesh.faces)-1)
    for i in range(len(mesh.faces)):
        face = mesh.faces[i]
        vertices = mesh.vertices[face]
        vertices = torch.cat((vertices, vertices[0].unsqueeze(0)), dim=0).numpy()
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
    
if DO_SDFS_FROM_MESH_SAMPLING_WITH_CONTACT_FILTERING:
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


pdb.set_trace()

# Prepare to load pre-saved data from a finished run.
run_name = 'test_004'
system = 'bundlesdf_cube'
storage_name = file_utils.assure_created(op.join(file_utils.RESULTS_DIR, system))

# Load the exported outputs from the experiment run.
output_dir = file_utils.geom_for_bsdf_dir(storage_name, run_name)
normal_forces = torch.load(op.join(output_dir, 'normal_forces.pt')).detach()
points = torch.load(op.join(output_dir, 'points.pt')).detach()
directions = torch.load(op.join(output_dir, 'directions.pt')).detach()

# Perform filtering via simple thresholding of normal forces.
filtered_pts, filtered_dirs = filter_pts_and_dirs(points, directions, normal_forces)

print(f'{points.shape=}, {directions.shape=}')
print(f'{filtered_pts.shape=}, {filtered_dirs.shape=}')
# pdb.set_trace()

###
# Test 1:  Can build set of points manually.
# point_set = Tensor([[0.6, 0, 0], [0.6, 1, 0], [0, 1, 0], [0, 0, 0],
#                     [0.6, 0, 1], [0.6, 1, 1], [0, 1, 1], [0, 0, 1]])
# mesh = create_mesh_from_set_of_points(point_set)
    
# # Test 2:  Can load a pre-trained deep support convex network.
# network = load_deep_support_convex_network(storage_name, run_name)
# mesh = create_mesh_from_deep_support(network)

# Test 3:  Can build set of points from the saved support points.
mesh = create_mesh_from_set_of_points(filtered_pts)
###

# Sample points and visualize them.
sample_points, sample_normals = sample_on_mesh(mesh, 100)
print(f'{sample_points.shape=}, {sample_normals.shape=}')
# visualize_sampled_points(mesh, sample_points, sample_normals)
# visualize_sampled_points(None, filtered_pts, filtered_dirs)



# Generate training data.
ps, sdfs, vs, sdf_bounds = generate_training_data(sample_points, sample_normals)
ps_, sdfs_, vs_, sdf_bounds_ = generate_training_data(filtered_pts, filtered_dirs)
ps = torch.cat((ps, ps_), dim=0)
sdfs = torch.cat((sdfs, sdfs_), dim=0)
vs = torch.cat((vs, vs_), dim=0)
sdf_bounds = torch.cat((sdf_bounds, sdf_bounds_), dim=0)

print(f'{ps.shape=},{sdfs.shape=},{vs.shape=},{sdf_bounds.shape=}')

# Visualize it.  Note:  can call this visualization function without providing
# the training data, and it will generate some for visualization purposes.
visualize_sdfs(filtered_pts, filtered_dirs, ps=ps, sdfs=sdfs, vs=vs,
               sdf_bounds=sdf_bounds)
# visualize(ps,sdfs)
# visualize(vs,sdf_bounds)

# torch.save(ps, os.path.join(output_dir, 'support_pts.pt'))
# torch.save(sdfs, os.path.join(output_dir, 'sdfs_from_cnets.pt'))
# torch.save(vs, os.path.join(output_dir, 'sampled_pts.pt'))
# torch.save(sdf_bounds, os.path.join(output_dir, 'sdf_bounds_from_cnets.pt'))
