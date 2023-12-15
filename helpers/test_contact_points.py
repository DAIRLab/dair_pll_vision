import numpy as np
import torch
import os 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_normal_forces(system=None):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = f"../normal_forces_{system}.npy"
    file_path = os.path.join(curr_dir, relative_path)
    normal_forces_array = np.load(file_path)
    print(normal_forces_array.shape)
    return normal_forces_array

def load_contact_points(system=None):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = f"../contact_points_{system}.npy"
    file_path = os.path.join(curr_dir, relative_path)
    contact_points_array = np.load(file_path)
    print(contact_points_array.shape)
    return contact_points_array

def load_pretrained_weights():
    state_dict = torch.load('ICNN_weights.pth')

    for layer_name, weights in state_dict.items():
        print(f"Layer: {layer_name}")
        print(f"Weights: {weights}")
        print()

def obj2npy(obj_path, npy_path):
    vertices = []
    with open(obj_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                _, x, y, z = line.split()
                vertices.append([float(x), float(y), float(z)])
    vertices = np.array(vertices)
    np.save(npy_path, vertices)

def visualize(data):
    x, y, z = data[:,0], data[:,1], data[:,2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.title('3D Point Visualization')
    # plt.show()
    plt.savefig('contact_points.png')

def filter_pts(normal_forces, points):
    threshold = 0.01
    mask = normal_forces > threshold
    filtered_pts = points[mask.squeeze()]
    visualize(filtered_pts)
    print(filtered_pts.shape)
    print('Pts visualized!')
    filter_pts_path = 'filtered_pts.npy'
    np.save(filter_pts_path, filtered_pts)

if __name__ == "__main__":
    system = 'cube'
    contact_pts = load_contact_points(system)
    normal_forces = load_normal_forces(system)
    filter_pts(normal_forces, contact_pts)
