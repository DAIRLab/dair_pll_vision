import numpy as np
from stl import mesh
from scipy.spatial import ConvexHull
# pip install scipy numpy-stl

def obj_to_stl(input_obj_file, output_stl_file):
    # Read the obj file
    vertices = []
    faces = []
    with open(input_obj_file, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertices.append(list(map(float, line.split()[1:])))
            elif line.startswith('f'):
                # Assuming that the .obj mesh is triangulated.
                faces.append(list(map(int, line.split()[1:])))

    vertices = np.array(vertices)
    faces = np.array(faces) - 1  # OBJ files use 1-indexing

    mesh_data = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            mesh_data.vectors[i][j] = vertices[face[j], :]

    mesh_data.save(output_stl_file)

def stl_to_obj(input_stl_file, output_obj_file):
    mesh_data = mesh.Mesh.from_file(input_stl_file)

    with open(output_obj_file, 'w') as file:
        for v in mesh_data.vectors.reshape((-1, 3)):
            file.write(f"v {' '.join(map(str, v))}\n")
        for i in range(0, len(mesh_data.vectors) * 3, 3):
            file.write(f"f {i+1} {i+2} {i+3}\n")

def create_convex_hull(input_obj_file, output_obj_file):
    # Convert obj to stl for easier handling
    obj_to_stl(input_obj_file, 'temp.stl')

    # Load STL and compute convex hull
    input_mesh = mesh.Mesh.from_file('temp.stl')
    points = input_mesh.vectors.reshape((-1, 3))
    hull = ConvexHull(points)

    # Create the convex hull mesh
    convex_mesh = mesh.Mesh(np.zeros(hull.simplices.shape[0], dtype=mesh.Mesh.dtype))
    for i, simplex in enumerate(hull.simplices):
        convex_mesh.vectors[i] = points[simplex]

    # Save convex hull as stl
    convex_mesh.save('temp_hull.stl')

    # Convert the convex hull stl to obj
    stl_to_obj('temp_hull.stl', output_obj_file)

if __name__ == "__main__":
    create_convex_hull('results/cube_hand_toss/mesh_cleaned.obj', 'results/cube_hand_toss/mesh_convex_hull.obj')
