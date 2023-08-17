from dair_pll.deep_support_function import extract_outward_normal_hyperplanes, get_mesh_summary_from_polygon
from dair_pll.geometry import Polygon
import pywavefront  # type: ignore
import torch
from torch import Tensor
torch.set_printoptions(threshold=10_000)
filename = '/home/cnets-vision/mengti_ws/dair_pll_latest/assets/cube_convex_hull_simplified.obj'
# filename = '/home/cnets-vision/mengti_ws/dair_pll_latest/assets/cube_hand_toss_optimized.obj'

mesh = pywavefront.Wavefront(filename)
vertices = Tensor(mesh.vertices)
polygon = Polygon(vertices)
mesh_summary = get_mesh_summary_from_polygon(polygon)
normals = extract_outward_normal_hyperplanes(
        mesh_summary.vertices.unsqueeze(0),
        mesh_summary.faces.unsqueeze(0)
    )[0].squeeze(0)
with open('cube_convex_hull_with_normals.obj', 'w') as file:
    # Write vertices to the file
    for vertex in vertices.numpy():
        file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

    # Write normals to the file
    for normal in normals.numpy():
        file.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")

    # Write faces to the file (assuming you have them)
    # This is just a basic example, adjust based on your faces' structure
    for face in mesh_summary.faces.numpy():
        # +1 because obj indexing starts at 1, not 0
        file.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")