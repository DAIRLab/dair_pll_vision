import numpy as np
import matplotlib.pyplot as plt
import trimesh
from scipy.spatial import ConvexHull
from itertools import combinations
import open3d as o3d

def get_inertia(obj_file):
    coords = np.loadtxt(obj_file, unpack=True, delimiter=',', dtype=int)
    coords = coords[:3, :]
    coords[[1, 2]] = coords[[2, 1]]
    
    coords = coords/max(coords.ravel())
    x, y, z  = coords

    x_mean, y_mean = np.mean(x), np.mean(y)
    z_max = max(z)
    P0 = x0, y0, z0 = x_mean, y_mean, z_max
    coords = coords.T - P0
    coords = coords.T
    x, y, z = coords
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x,y,z)
    ax.view_init(elev=20)
    N = coords.shape[1]
    Ix = sum(coords[1]**2 + coords[2]**2)/N
    Iy = sum(coords[0]**2 + coords[2]**2)/N
    Iz = sum(coords[0]**2 + coords[1]**2)/N
    Ixy = sum(coords[0]*coords[1])/N
    Iyz = sum(coords[1]*coords[2])/N
    Ixz = sum(coords[0]*coords[2])/N
    return np.array([[Ix, Ixy, Ixz],[Ixy, Iy, Iyz],[Ixz, Iyz, Iz]])

def ensure_vertex_normals(mesh):
    # Accessing vertex_normals will compute them if they aren't already present
    mesh.vertex_normals = (mesh.vertex_normals.T / np.linalg.norm(mesh.vertex_normals, axis=1)).T
    return mesh

def simplify_mesh(input_path, output_path, fraction):
    # Load the mesh from the given .obj file
    mesh = trimesh.load_mesh(input_path)

    # Simplify the mesh
    mesh_simplified = mesh.simplify_quadratic_decimation(int(fraction * len(mesh.faces)))
    # mesh_simplified = mesh.simplify_quadratic_decimation(10)
    # Export the simplified mesh to an .obj file
    mesh_simplified.export(output_path)

def simplify_to_cube(input_path, output_path):
    # 1. Load the dense mesh
    mesh = trimesh.load_mesh(input_path)
    
    # 2. Compute the axis-aligned bounding box of the mesh
    min_bound = np.min(mesh.vertices, axis=0)
    max_bound = np.max(mesh.vertices, axis=0)
    
    # 3. Determine the side length of the cube using the average dimension of the AABB
    dimensions = max_bound - min_bound
    side_length = np.mean(dimensions)
    half_length = side_length / 2
    
    # 4. Create cube vertices with equal side lengths centered around the mesh's centroid
    centroid = mesh.centroid
    cube_vertices = [
        [centroid[0] - half_length, centroid[1] - half_length, centroid[2] - half_length],
        [centroid[0] + half_length, centroid[1] - half_length, centroid[2] - half_length],
        [centroid[0] - half_length, centroid[1] + half_length, centroid[2] - half_length],
        [centroid[0] + half_length, centroid[1] + half_length, centroid[2] - half_length],
        [centroid[0] - half_length, centroid[1] + half_length, centroid[2] + half_length],
        [centroid[0] + half_length, centroid[1] + half_length, centroid[2] + half_length],
        [centroid[0] - half_length, centroid[1] - half_length, centroid[2] + half_length],
        [centroid[0] + half_length, centroid[1] - half_length, centroid[2] + half_length]
    ]

    # Construct faces for a cube
    cube_faces = [
        [0, 1, 2], [2, 1, 3],
        [4, 5, 6], [6, 5, 7],
        [0, 1, 6], [6, 1, 7],
        [2, 3, 4], [4, 3, 5],
        [1, 3, 5], [1, 5, 7],
        [0, 2, 4], [0, 4, 6]
    ]

    # 5. Create a mesh with the cube vertices and faces
    cube = trimesh.Trimesh(vertices=cube_vertices, faces=cube_faces)

    # 6. Save the cube mesh
    cube.export(output_path)

def add_normals_to_obj(input_path, output_path):
    # Load the mesh from the given file
    mesh = trimesh.load(input_path)
    # Compute the vertex normals
    mesh_with_normals = ensure_vertex_normals(mesh)
    # Save the mesh with normals back to .obj format
    print(mesh_with_normals.vertices.dtype)
    mesh_with_normals.export(output_path, file_type="obj")
    print(f'mesh exported to {output_path}')

def maximal_volume_subset(mesh_path, vertex_count=20):
    mesh = trimesh.load_mesh(mesh_path)
    vertices = mesh.vertices

    max_volume = -np.inf
    best_subset = None

    # Iterate over all combinations of vertices
    for subset in combinations(vertices, vertex_count):
        hull = ConvexHull(np.array(subset))
        if hull.volume > max_volume:
            max_volume = hull.volume
            best_subset = subset

    return np.array(best_subset)

def create_max_volume_obj(input_path, output_path, vertex_count=20, target_vertex_count=None, target_face_count=None):
    # Get the best subset of vertices
    subset = maximal_volume_subset(input_path, vertex_count=vertex_count)
    
    # Create a convex hull from the subset
    hull = ConvexHull(subset)

    # Convert the convex hull to a Trimesh object
    mesh = trimesh.Trimesh(vertices=hull.points, faces=hull.simplices)
    while len(mesh.vertices) > target_vertex_count or len(mesh.faces) > target_face_count:
        mesh = mesh.simplify_quadratic_decimation(int(0.98 * len(mesh.faces)))

        # Safety check to avoid infinite loops
        if len(mesh.faces) <= 1:
            print("Cannot achieve target vertex and face count while preserving connectivity.")
            break
    # Export the new mesh to an .obj file
    mesh.export(output_path)
    print(f'max-vol mesh exported to {output_path}')

def mesh_to_cube(file_path, output_path):
    # Load the mesh using trimesh
    mesh = trimesh.load(file_path)

    # Determine the maximal magnitude among all vertices of the mesh
    max_magnitude = np.max(np.abs(mesh.vertices)) * 0.05

    # Define vertices for the cube
    cube_vertices = [
        [-max_magnitude, -max_magnitude, -max_magnitude],
        [ max_magnitude, -max_magnitude, -max_magnitude],
        [ max_magnitude,  max_magnitude, -max_magnitude],
        [-max_magnitude,  max_magnitude, -max_magnitude],
        [-max_magnitude, -max_magnitude,  max_magnitude],
        [ max_magnitude, -max_magnitude,  max_magnitude],
        [ max_magnitude,  max_magnitude,  max_magnitude],
        [-max_magnitude,  max_magnitude,  max_magnitude]
    ]

    # Faces for the cube
    cube_faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7]
    ]
    cube = trimesh.Trimesh(vertices=cube_vertices, faces=cube_faces)
    cube.export(output_path)

def shrink_mesh_file(mesh_file_path, divisor, output_file_path=None):
    # Read the input mesh file
    with open(mesh_file_path, 'r') as f:
        lines = f.readlines()

    # Placeholder for the shrunken lines
    shrunken_lines = []

    for line in lines:
        # Split the line into parts
        parts = line.split()

        # Check if this line represents a vertex
        if len(parts) > 0 and parts[0] == "v":
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            x, y, z = x / divisor, y / divisor, z / divisor
            shrunken_lines.append(f"v {x} {y} {z}\n")
        else:
            shrunken_lines.append(line)

    # Write the modified mesh to the output file
    with open(output_file_path, 'w') as f:
        f.writelines(shrunken_lines)

    print(f"Shrunken mesh saved to {output_file_path}")

if __name__ == '__main__':
    # get inertia 
    # mesh_file = './mesh_convex_hull.txt'
    # print(get_inertia(mesh_file))
   
    original_mesh = '/home/cnets-vision/mengti_ws/dair_pll_latest/assets/cube_convex_hull.obj'
    simplified_mesh = '/home/cnets-vision/mengti_ws/dair_pll_latest/assets/cube_convex_hull_rescale_simplified.obj'
    output_path = '/home/cnets-vision/mengti_ws/dair_pll_latest/assets/cube_convex_hull_optimized.obj'
    normal_mesh = '/home/cnets-vision/mengti_ws/dair_pll_latest/assets/cube_convex_hull_with_normals.obj'
    rescale_mesh = '/home/cnets-vision/mengti_ws/dair_pll_latest/assets/cube_convex_hull_rescale.obj'
    alt_simplified_mesh = '/home/cnets-vision/mengti_ws/dair_pll_latest/assets/cube_convex_hull_rescale_simplified_alt.obj'
    # simplify_mesh(original_mesh, simplified_mesh, 0.01)
    # create_max_volume_obj(simplified_mesh, output_path, vertex_count=10, target_vertex_count=8, target_face_count=6)
    # add_normals_to_obj(output_path, normal_mesh)
    # mesh_to_cube(original_mesh, output_path)
    # add_normals_to_obj(simplified_mesh, normal_mesh)
    
    
    # shrink_mesh_file(original_mesh, 8.7, rescale_mesh)
    # simplify_mesh(rescale_mesh, simplified_mesh, 0.01)
    
    simplify_to_cube(rescale_mesh, alt_simplified_mesh)


