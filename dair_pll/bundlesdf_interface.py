"""Functionality related to connecting ContactNets to BundleSDF."""

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from dair_pll import file_utils
from dair_pll.file_utils import geom_for_bsdf_dir
from examples.contactnets_simple import DATA_ASSETS
import torch
from torch import Tensor

import pdb
import click


# Hyperparameters for querying into and outside of the object at an SDF=0 point.
AXIS_NEARBY_DEPTH = 0.005
AXIS_OUTSIDE_DEPTH = 0.1
AXIS_NEARBY_N_QUERY = 50     # will do twice this:  once inside, once outside
AXIS_OUTSIDE_N_QUERY = 50

# Hyperparameters for querying around an object with SDF minimum bounds.
BOUNDED_NEARBY_DEPTH = 0.005
BOUNDED_NEARBY_RADIUS = 0.05
BOUNDED_FAR_DEPTH = 0.1
BOUNDED_FAR_RADIUS = 0.1
BOUNDED_NEARBY_N_QUERY = 100
BOUNDED_FAR_N_QUERY = 100

# Hyperparameters for filtering support points
FORCE_THRES = 0.3676 #N

def generate_point_sdf_pairs(points: Tensor, directions: Tensor
                             ) -> Tuple[Tensor, Tensor]:
    """Generate pairs of 3D points and their associated signed distance, given
    a list of points with known signed distance of zero and directions
    associated with those points' contact directions.

    Args:
        point (M, 3):  M support points of the object geometry for the given
            support directions.
        direction (M, 3):  associated support directions.

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
        -AXIS_NEARBY_DEPTH*torch.ones(AXIS_NEARBY_N_QUERY),
        AXIS_NEARBY_DEPTH*torch.ones(AXIS_NEARBY_N_QUERY),
        AXIS_OUTSIDE_DEPTH*torch.ones(AXIS_OUTSIDE_N_QUERY)
    )).repeat(n_points)
    signed_distances = distance_scalings * torch.rand_like(distance_scalings)

    n_per_point = 2*AXIS_NEARBY_N_QUERY + AXIS_OUTSIDE_N_QUERY

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
    n_per_point = 2*BOUNDED_NEARBY_N_QUERY + BOUNDED_FAR_N_QUERY
    radius_scalings = torch.cat((
        BOUNDED_NEARBY_RADIUS*torch.ones(2*BOUNDED_NEARBY_N_QUERY),
        BOUNDED_FAR_RADIUS*torch.ones(BOUNDED_FAR_N_QUERY)
    )).repeat(n_points)
    height_scalings = torch.cat((
        -BOUNDED_NEARBY_DEPTH*torch.ones(BOUNDED_NEARBY_N_QUERY),
        BOUNDED_NEARBY_DEPTH*torch.ones(BOUNDED_NEARBY_N_QUERY),
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
    ax.scatter(vs[:, 0], vs[:, 1], vs[:, 2], c=sdf_bounds, cmap='viridis',
               marker='.', label='Points with SDF bound')

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

def filter_pts_and_dirs(contact_points, directions, normal_forces):
    """Filter out points that are not exactly in contact with the ground.
    """
    assert normal_forces.ndim == 1
    assert contact_points.ndim == directions.ndim == 2
    assert normal_forces.shape[0] == contact_points.shape[0] == directions.shape[0]
    assert contact_points.shape[1] == directions.shape[1] == 3
    mask = normal_forces > FORCE_THRES
    filtered_points = contact_points[mask]
    filtered_directions = directions[mask]
    return filtered_directions.detach(), filtered_points.detach()


run_name = 'test_003'
system = 'bundlesdf_cube'
data_asset = DATA_ASSETS[system]
storage_name = file_utils.assure_created(
        os.path.join(file_utils.RESULTS_DIR, data_asset)
    )
output_dir = geom_for_bsdf_dir(storage_name, run_name)
normal_forces = torch.load(os.path.join(output_dir, 'normal_forces.pt'))
points = torch.load(os.path.join(output_dir, 'points.pt'))
directions = torch.load(os.path.join(output_dir, 'directions.pt'))
print(f'{points.shape=}, {directions.shape=}')
filterted_dirs, filterted_pts = filter_pts_and_dirs(points, directions, normal_forces)
print(f'{filterted_pts.shape=}, {filterted_dirs.shape=}')

# Generate training data.
ps, sdfs, vs, sdf_bounds = generate_training_data(filterted_pts, filterted_dirs)
print(f'{ps.shape=},{sdfs.shape=},{vs.shape=},{sdf_bounds.shape=}')

# Visualize it.  Note:  can call this visualization function without providing
# the training data, and it will generate some for visualization purposes.
visualize_sdfs(filterted_pts, filterted_dirs, ps=ps, sdfs=sdfs, vs=vs,
               sdf_bounds=sdf_bounds)


torch.save(ps, os.path.join(output_dir, 'support_pts.pt'))
torch.save(sdfs, os.path.join(output_dir, 'sdfs_from_cnets.pt'))
torch.save(vs, os.path.join(output_dir, 'sampled_pts.pt'))
torch.save(sdf_bounds, os.path.join(output_dir, 'sdf_bounds_from_cnets.pt'))