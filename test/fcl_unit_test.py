"""Test script for FCL debugging.

(Pdb) p_WoCo_W
tensor([[[ 0.5251, -0.0066,  0.0648],    # object
         [ 0.5505, -0.0008,  0.4956],    # robot
         [ 0.0000,  0.0000,  0.0000]]])  # ground

(Pdb) R_WC
tensor([[[[ 0.4586,  0.1976, -0.8664],
          [ 0.8887, -0.0981,  0.4480],
          [ 0.0035, -0.9754, -0.2206]],

         [[ 1.0000,  0.0062, -0.0021],
          [-0.0061,  0.9998,  0.0202],
          [ 0.0022, -0.0202,  0.9998]],

         [[ 1.0000,  0.0000,  0.0000],
          [ 0.0000,  1.0000,  0.0000],
          [ 0.0000,  0.0000,  1.0000]]]])

(Pdb) print(q)
tensor([[-0.0736, -0.0224,  0.0793, -1.6270, -0.0185,  1.6067,  0.0125,  0.5338,
         -0.6666, -0.4074,  0.3236,  0.5251, -0.0066,  0.0648]])

DSF output saved at test/test_obj.obj.
Robot EE is a sphere of radius 0.0195.

Results with python-fcl 0.7.0.4 and 0.7.0.6 exactly match:
    Collision: 0
    Distance: 0.34939163307063303

    Mesh point closest to sphere: [-0.00818778 -0.06070913 -0.01783965]
    Magnitude: 0.0638035378426912

    Sphere point closest to mesh: [-0.00139599 -0.00041431 -0.01944555]
    Magnitude: 0.0195

    Direction: [ 0.00679179  0.06029482 -0.0016059 ]
    Magnitude: 0.060697383676719154

    Distance result `o1`: <fcl.fcl.BVHModel object at 0x7feab70c0a70>
    Distance result `o2`: <fcl.fcl.Sphere object at 0x7feb8630e970>

"""

import fcl
import numpy as np
import pdb
import torch
import trimesh

from dair_pll.tensor_utils import pbmm
from dair_pll.geometry import fcl_distance_nearest_points_cleaner


def print_distance_result(o1_name, o2_name, result):
    print(f"Distance between {o1_name} and {o2_name}:")
    print("-" * 30)
    print(f"Distance: {result.min_distance}")
    print("Closest Points:")
    print(result.nearest_points[0])
    print(result.nearest_points[1])
    print(f"Distance between closest points:")
    print(np.linalg.norm(result.nearest_points[0] - result.nearest_points[1]))
    print("")


SPHERE_RADIUS = 0.0195
MESH_FILEPATH = 'test_obj.obj'

p_WoCo_W = torch.tensor([[[ 0.5251, -0.0066,  0.0648],    # object
                          [ 0.5505, -0.0008,  0.4956],    # robot
                          [ 0.0000,  0.0000,  0.0000]]])  # ground
R_WC = torch.tensor([[[[ 0.4586,  0.1976, -0.8664],       # object
                       [ 0.8887, -0.0981,  0.4480],
                       [ 0.0035, -0.9754, -0.2206]],
                      [[ 1.0000,  0.0062, -0.0021],       # robot
                       [-0.0061,  0.9998,  0.0202],
                       [ 0.0022, -0.0202,  0.9998]],
                      [[ 1.0000,  0.0000,  0.0000],       # ground
                       [ 0.0000,  1.0000,  0.0000],
                       [ 0.0000,  0.0000,  1.0000]]]])


# Build the sphere FCL geometry.
sphere_fcl_geometry = fcl.Sphere(SPHERE_RADIUS)

# Load the mesh and build the mesh FCL geometry.
trimesh_mesh = trimesh.load(MESH_FILEPATH)
vertices = np.array(trimesh_mesh.vertices)   # Same result if do np.array or not
faces = np.array(trimesh_mesh.faces)

mesh_fcl_geometry = fcl.BVHModel()
mesh_fcl_geometry.beginModel(vertices.shape[0], faces.shape[0])
mesh_fcl_geometry.addSubModel(vertices, faces)
mesh_fcl_geometry.endModel()

# Creating a convex FCL geometry from the vertices produces a seg fault.
# flat_faces = np.concatenate((3 * np.ones((len(faces), 1), dtype=np.int64),
#                              faces), axis=1).flatten()
# convex_fcl_geometry = fcl.Convex(vertices, len(flat_faces), flat_faces)

# Get the transformation between the object (B) and sphere (A).
R_WA = R_WC[..., 1, :, :]
R_WB = R_WC[..., 0, :, :]
R_AB = pbmm(R_WA.transpose(-1, -2), R_WB).reshape(3, 3)
p_AoBo_W = p_WoCo_W[:, 0, :] - p_WoCo_W[:, 1, :]
p_AoBo_A = pbmm(p_AoBo_W.unsqueeze(-2), R_WA).squeeze(-2).reshape(3)

# Build FCL collision objects.
b_t = fcl.Transform(R_AB.detach().numpy(), p_AoBo_A.detach().numpy())
a_obj = fcl.CollisionObject(sphere_fcl_geometry, fcl.Transform())
b_obj = fcl.CollisionObject(mesh_fcl_geometry, b_t)

# Do collision checking.
collision_request = fcl.CollisionRequest()
collision_request.enable_contact = True
distance_request = fcl.DistanceRequest()
distance_request.enable_nearest_points = True
distance_request.enable_signed_distance = True  # Doesn't matter.

collision_result = fcl.CollisionResult()
print(f'Collision: {fcl.collide(a_obj, b_obj, collision_request, collision_result)}')

distance_result = fcl.DistanceResult()
distance = fcl.distance(a_obj, b_obj, distance_request, distance_result)
print(f'Distance: {distance}')
print_distance_result('Sphere', 'Mesh', distance_result)


# Do some checks.
mesh_pt_closest_to_sphere, sphere_pt_closest_to_mesh = \
    distance_result.nearest_points
print(f'\nMesh point closest to sphere: {mesh_pt_closest_to_sphere}')
print(f'Magnitude: {np.linalg.norm(mesh_pt_closest_to_sphere)}')
print(f'\nSphere point closest to mesh: {sphere_pt_closest_to_mesh}')
print(f'Magnitude: {np.linalg.norm(sphere_pt_closest_to_mesh)}')

direction = sphere_pt_closest_to_mesh - mesh_pt_closest_to_sphere
print(f'\nDirection: {direction}')
print(f'Magnitude: {np.linalg.norm(direction)}')

print(f'\nDistance result `o1`: {distance_result.o1}')
print(f'Distance result `o2`: {distance_result.o2}')

# Now it actually seems like the witness point on the mesh to the sphere is
# represented in the mesh's frame.
sphere_index = 0 if distance_result.o1 == sphere_fcl_geometry else 1
mesh_index = 0 if distance_result.o1 == mesh_fcl_geometry else 1
assert sphere_index != mesh_index

sphere_pt_A = distance_result.nearest_points[sphere_index]
mesh_pt_B = distance_result.nearest_points[mesh_index]

assert np.isclose(np.linalg.norm(sphere_pt_A), SPHERE_RADIUS)

mesh_pt_A = (pbmm(
    torch.tensor(mesh_pt_B, dtype=torch.float64),
    torch.tensor(R_AB, dtype=torch.float64).T) + p_AoBo_A).numpy()
nearest_point_distance = np.linalg.norm(mesh_pt_A - sphere_pt_A)
print(f'\nDistance w/ converted closest points: {nearest_point_distance}')
print(f'Distance reported: {distance}')
print(f'Discrepancy: {nearest_point_distance - distance}\n')


# Check the cleaning function.
distance_result = fcl.DistanceResult()
sphere_pt_sphere, mesh_pt_sphere = fcl_distance_nearest_points_cleaner(
    a_geom=sphere_fcl_geometry,
    b_geom=mesh_fcl_geometry,
    a_obj=a_obj,
    b_obj=b_obj,
    distance_request=distance_request,
    result=distance_result)




pdb.set_trace()

