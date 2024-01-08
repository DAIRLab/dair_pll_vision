"""Functionality related to connecting ContactNets to BundleSDF."""

from typing import Tuple
import torch
from torch import Tensor
import pdb


# Hyperparameters for querying into and outside of the object at an SDF=0 point.
OUTSIDE_DEPTH = 0.1
NEARBY_DEPTH = 0.005
N_QUERY_OUTSIDE_AXIS = 5
N_QUERY_NEARBY = 5

# Hyperparameters for querying around an object with SDF minimum bounds.
N_QUERY_ = 100


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
        -NEARBY_DEPTH*torch.rand(N_QUERY_NEARBY),
        NEARBY_DEPTH*torch.rand(N_QUERY_NEARBY),
        OUTSIDE_DEPTH*torch.rand(N_QUERY_OUTSIDE_AXIS)
    ))

    n_points = 2*N_QUERY_NEARBY + N_QUERY_OUTSIDE_AXIS
    
    repeated_point = point.repeat((n_points, 1))
    repeated_direction = direction.repeat((n_points, 1))

    # Get the 3D points corresponding to the generated signed distances.
    points_on_axis = repeated_point + \
        signed_distances.unsqueeze(1).repeat(1,3)*repeated_direction

    return points_on_axis, signed_distances


# points = torch.load('./points.pt')
# directions = torch.load('./directions.pt')
points = torch.Tensor([[1.2, 0.8, 1.0]])
directions = torch.Tensor([[1., 0., 0.]])

print(f'{points.shape=}, {directions.shape=}')

ps, sdfs = generate_point_sdf_pairs_from_point_direction(points[0], directions[0])

pdb.set_trace()
