"""Functionality related to connecting ContactNets to BundleSDF."""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

import pdb


# Hyperparameters for querying into and outside of the object at an SDF=0 point.
AXIS_NEARBY_DEPTH = 0.005
AXIS_OUTSIDE_DEPTH = 0.1
AXIS_NEARBY_N_QUERY = 50     # will do twice this:  once inside, once outside
AXIS_OUTSIDE_N_QUERY = 50

# Hyperparameters for querying around an object with SDF minimum bounds.
BOUNDED_NEARBY_DEPTH = 0.005
BOUNDED_NEARBY_RADIUS = 0.1
BOUNDED_FAR_DEPTH = 0.1
BOUNDED_FAR_RADIUS = 0.2
BOUNDED_NEARBY_N_QUERY = 100
BOUNDED_FAR_N_QUERY = 100


def generate_point_sdf_pairs_from_point_direction(
        point: Tensor, direction: Tensor) -> Tuple[Tensor, Tensor]:
    """Generate pairs of 3D points and their associated signed distance, given
    a point with known signed distance of zero and a direction associated with
    that point's contact direction.

    Args:
        point (3,):  support point of the object geometry for the given support
            direction.
        direction (3,):  support direction.

    Outputs:
        points_on_axis (N, 3):  N 3D points, generated along the ray passing
            through the provided point in the provided direction.
        signed_distances (N,):  N signed distances associated with the points.
    """
    assert point.shape == (3,), f'Expected {point.shape=} to be (3,).'
    assert direction.shape == (3,), f'Expected {direction.shape=} to be (3,).'
    
    # Randomly sample distances inside and outside the object geometry at the
    # provided support point.
    signed_distances = torch.cat((
        -AXIS_NEARBY_DEPTH*torch.rand(AXIS_NEARBY_N_QUERY),
        AXIS_NEARBY_DEPTH*torch.rand(AXIS_NEARBY_N_QUERY),
        AXIS_OUTSIDE_DEPTH*torch.rand(AXIS_OUTSIDE_N_QUERY)
    ))

    n_points = 2*AXIS_NEARBY_N_QUERY + AXIS_OUTSIDE_N_QUERY
    
    repeated_point = point.repeat((n_points, 1))
    repeated_direction = direction.repeat((n_points, 1))

    # Get the 3D points corresponding to the generated signed distances.
    points_on_axis = repeated_point + \
        signed_distances.unsqueeze(1).repeat(1,3)*repeated_direction

    return points_on_axis, signed_distances


def generate_point_sdf_bound_pairs_from_point_direction(
        point: Tensor, direction: Tensor) -> Tuple[Tensor, Tensor]:
    """Generate pairs of 3D points and their associated minimum signed distance
    bound, given a point with known signed distance of zero and a direction
    associated with that point's contact direction.

    Args:
        point (3,):  support point of the object geometry for the given support
            direction.
        direction (3,):  support direction.

    Outputs:
        points_in_space (N, 3):  N 3D points, generated randomly.
        min_signed_distances (N,):  N minimum signed distance bounds associated
            with the points.
    """
    assert point.shape == (3,), f'Expected {point.shape=} to be (3,).'
    assert direction.shape == (3,), f'Expected {direction.shape=} to be (3,).'

    # Get one unit vector orthogonal to the provided direction vector.  Can use
    # an intermediate random vector, then cross with `direction` to get an
    # orthogonal one.
    orth_dir_x = torch.nn.functional.normalize(
        torch.cross(direction, torch.rand(3)), dim=0
    )

    # Get the third unit vector to complete the coordinate system, where the z
    # axis is represented by the input `direction``.
    orth_dir_y = torch.nn.functional.normalize(
        torch.cross(direction, orth_dir_x), dim=0
    )

    # Randomly sample radii, heights, and angles.
    radii = torch.cat((
        BOUNDED_NEARBY_RADIUS*torch.rand(2*BOUNDED_NEARBY_N_QUERY),
        BOUNDED_FAR_RADIUS*torch.rand(BOUNDED_FAR_N_QUERY)
    ))
    heights = torch.cat((
        -BOUNDED_NEARBY_DEPTH*torch.rand(BOUNDED_NEARBY_N_QUERY),
        BOUNDED_NEARBY_DEPTH*torch.rand(BOUNDED_NEARBY_N_QUERY),
        BOUNDED_FAR_DEPTH*torch.rand(BOUNDED_FAR_N_QUERY)
    ))
    n_points = 2*BOUNDED_NEARBY_N_QUERY + BOUNDED_FAR_N_QUERY
    angles = 2*torch.pi*torch.rand(n_points)
    
    repeated_point = point.repeat((n_points, 1))
    repeated_z = direction.repeat((n_points, 1))
    repeated_x = orth_dir_x.repeat((n_points, 1))
    repeated_y = orth_dir_y.repeat((n_points, 1))
    repeated_radii = radii.unsqueeze(1).repeat(1,3)
    repeated_heights = heights.unsqueeze(1).repeat(1,3)
    repeated_angles = angles.unsqueeze(1).repeat(1,3)

    # Construct the 3D points from the radii, heights, and angles.
    points_in_space = repeated_point + \
        repeated_heights*repeated_z + \
        repeated_radii*torch.cos(repeated_angles)*repeated_x + \
        repeated_radii*torch.sin(repeated_angles)*repeated_y

    min_signed_distances = heights

    return points_in_space, min_signed_distances


def visualize_sdfs(point: Tensor, direction: Tensor) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the original support point and direction.
    ax.plot(*point, marker='*', markersize=20, color='r',
            label='support point')
    ax.quiver(*point, *direction, color='r', label='support direction')

    # Generate SDF points along axis.
    ps, sdfs = generate_point_sdf_pairs_from_point_direction(point, direction)
    colored_sdfs = ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], c=sdfs,
                              cmap='viridis', marker='o')
    
    # Generate more bounded SDFs.
    vs, sdf_bounds = generate_point_sdf_bound_pairs_from_point_direction(
        point, direction)
    ax.scatter(vs[:, 0], vs[:, 1], vs[:, 2], c=sdf_bounds, cmap='viridis',
               marker='.')

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


# points = torch.load('./points.pt')
# directions = torch.load('./directions.pt')
points = torch.Tensor([[1.2, 0.8, 1.0]])
directions = torch.Tensor([[1., 0., 0.]])

print(f'{points.shape=}, {directions.shape=}')

# ps, sdfs = generate_point_sdf_pairs_from_point_direction(points[0], directions[0])
# vs, sdf_bounds = generate_point_sdf_bound_pairs_from_point_direction(points[0], directions[0])

visualize_sdfs(points[0], directions[0])

pdb.set_trace()
