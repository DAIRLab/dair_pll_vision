import numpy as np
import trimesh
from dair_pll.deep_support_function import extract_obj
from dair_pll.geometry import HomogeneousICNN
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Parameter
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

_DEEP_SUPPORT_DEFAULT_DEPTH = 2
_DEEP_SUPPORT_DEFAULT_WIDTH = 256
LR = 1e-3
EPOCHS = 200
NUM_SAMPLES = 2000 #20000
OUTPUT_DIR = './'

def fibonacci_sphere_samples(samples):
    points = []
    phi = torch.pi * (3. - torch.sqrt(torch.tensor(5.)))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        y_tensor = torch.tensor(y, dtype=torch.float32)
        radius = torch.sqrt(1 - y_tensor * y_tensor)  # radius at y

        theta = phi * i  # golden angle increment

        x = torch.cos(theta) * radius
        z = torch.sin(theta) * radius

        points.append(torch.tensor([x, y, z]))
    return torch.stack(points)

def is_point_in_triangle(vertices, point):
    # Unpack the vertices A, B, and C
    A, B, C = vertices

    # Vectors from A to B and A to C
    v0 = B - A
    v1 = C - A
    v2 = point - A

    # Compute dot products
    dot00 = torch.dot(v0, v0)
    dot01 = torch.dot(v0, v1)
    dot02 = torch.dot(v0, v2)
    dot11 = torch.dot(v1, v1)
    dot12 = torch.dot(v1, v2)

    # Compute barycentric coordinates
    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

    # Check if point is in triangle
    return (u >= 0) and (v >= 0) and (u + v < 1)

# def sample_directions_and_intersection_points(path, num_samples):
#     mesh = trimesh.load(path, force='mesh')
#     vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
#     faces = mesh.faces
#     surface_normals = torch.tensor(mesh.face_normals, dtype=torch.float32)
#     directions, pts = [], []

#     sampled_directions = fibonacci_sphere_samples(num_samples)
#     for direction in sampled_directions:
#         direction = torch.tensor(direction, dtype=torch.float32)
#         direction /= torch.norm(direction)

#         for face_index, face_normal in zip(faces, surface_normals):
#             face_vertices = vertices[face_index]
            
#             distance_to_plane = torch.dot(face_normal, face_vertices[0])
#             denominator = torch.dot(face_normal, direction)
#             if denominator == 0:
#                 continue
#             t = (distance_to_plane - torch.dot(face_normal, torch.tensor([0.0, 0.0, 0.0]))) / denominator
#             intersection_point = torch.tensor([0.0, 0.0, 0.0]) + t * direction
#             if is_point_in_triangle(face_vertices, intersection_point):
#                 directions.append(direction)
#                 pts.append(intersection_point)

#     directions = torch.stack(directions)
#     pts = torch.stack(pts)
#     return directions, pts

def find_support_point(mesh, direction):
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    dot_products = torch.matmul(vertices, direction)
    max_index = torch.argmax(dot_products)
    return vertices[max_index]


def sample_directions_and_support_points(mesh_path, num_samples):
    mesh = trimesh.load(mesh_path)
    directions = fibonacci_sphere_samples(num_samples)
    support_points = [find_support_point(mesh, dir) for dir in directions]
    support_points = torch.stack(support_points)
    return directions, support_points


def pretrain_icnn(path, gt_dirs, gt_pts):
    mesh = trimesh.load(path)
    vertices = torch.tensor(mesh.vertices , dtype=torch.float32)

    length_scale = (vertices.max(dim=0).values -
                        vertices.min(dim=0).values).norm() / 2
    network = HomogeneousICNN(_DEEP_SUPPORT_DEFAULT_DEPTH, _DEEP_SUPPORT_DEFAULT_WIDTH, scale=length_scale)
    optimizer = optim.Adam(network.parameters(), lr=LR)
    min_loss = float('inf')
    best_state = None
    loss_function = nn.MSELoss()

    for epoch in range(EPOCHS):
        total_loss = 0

        optimizer.zero_grad()
        pred_vertices = network(gt_dirs)
        pred_support = (gt_dirs * pred_vertices).sum(dim=1)
        gt_support = (gt_dirs * gt_pts).sum(dim=1)
        # print(pred_support, gt_support)
        loss = loss_function(pred_support, gt_support)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if total_loss < min_loss:
            min_loss = total_loss
            best_state = network.state_dict()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(gt_dirs)}")
    
    network.load_state_dict(best_state)

    mesh_name = "test.obj"
    mesh_path = os.path.join(OUTPUT_DIR, mesh_name)
    with open(mesh_path, 'w', encoding="utf8") as new_obj_file:
        new_obj_file.write(extract_obj(network))

def visualize_dirs_and_pts(directions, support_points):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(support_points[:, 0], support_points[:, 1], support_points[:, 2], color='b', s=20, label='Support Points')
    origin = [0, 0, 0]
    for i in range(len(directions)):
        ax.quiver(*origin, *directions[i], length=np.linalg.norm(support_points[i]), arrow_length_ratio=0.1, color='r')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Visualization of Directions and Support Points')
    ax.legend()
    plt.show()

def plot_directions(directions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for d in directions:
        ax.quiver(0, 0, 0, d[0], d[1], d[2], length=1)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.show()

def plot_points(support_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(support_points[:, 0], support_points[:, 1], support_points[:, 2], color='b', s=20, label='Support Points')
    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([-0.1, 0.1])
    plt.show()

if __name__ == '__main__':
    path = './assets/mesh_cn_run_and_refined_convex_hull_with_normals.obj'
    directions, pts = sample_directions_and_support_points(path, num_samples=NUM_SAMPLES)
    # print(directions.size(), pts.size())
    # visualize_dirs_and_pts(directions, pts)
    # plot_directions(directions)
    # plot_points(pts)
    # print(pts)
    pretrain_icnn(path, directions, pts)


