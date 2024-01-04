import numpy as np
import torch
import os 
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_normal_forces():
    """
    Normal forces has shape (N, batch_size, 1)
    """
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = "../normal_forces_cube.npy"
    file_path = os.path.join(curr_dir, relative_path)
    normal_forces_array = np.load(file_path)
    if normal_forces_array.shape[0]>=256:
        np.save('normal_forces_save.npy', normal_forces_array)
        print('Saved normal forces')
    print(normal_forces_array.shape)

def load_contact_points(path):
    """
    Contact points has shape (N, batch_size, 3)
    """
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = "../contact_points_cube.npy"
    file_path = os.path.join(curr_dir, relative_path)
    contact_points_array = np.load(file_path)
    if contact_points_array.shape[0]>=256:
        np.save('contact_pts_save.npy', contact_points_array)
        print('Saved contact points')
    print(contact_points_array.shape)

def load_pretrained_weights(path):
    state_dict = torch.load(path)

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

def filter_pts(contact_points, normal_forces):
    '''
    Filter out points that are not in contact with the ground.
    '''
    flattened_forces = normal_forces.flatten()
    mean = np.mean(flattened_forces)
    std = np.std(flattened_forces)
    thres = mean + 1 * std
    print(f'thres: {thres}')
    normal_forces_squeezed = np.squeeze(normal_forces)
    mask = normal_forces_squeezed > thres
    filtered_points = contact_points[mask]
    flattened_filtered_points = filtered_points.reshape(-1, 3)
    return flattened_filtered_points

def visualize_and_filter(pts_path, force_path, img_path):
    contact_points = np.load(pts_path)
    normal_forces = np.load(force_path)
    # force_threshold = 0.05
    flattened_filtered_points = filter_pts(contact_points, normal_forces)
    print(f'filtered_pts: {flattened_filtered_points.shape}')
    np.save('filtered_contact_points.npy', flattened_filtered_points)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(flattened_filtered_points[:, 0], flattened_filtered_points[:, 1], flattened_filtered_points[:, 2])

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.savefig(img_path)

def visualize(contact_points, img_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(contact_points[:, 0], contact_points[:, 1], contact_points[:, 2])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.savefig(img_path)

def combine_pts(path):
    '''
    Combine contact points arrays to a single .npy
    '''
    file_count = 0
    max_number = 0
    
    for filename in os.listdir(path):
        if filename.endswith('.npy'):
            match = re.search(r'_(\d+)\.npy$', filename)
            if match:
                number = int(match.group(1))
                file_count += 1
                max_number = max(max_number, number)
    pts_arrays = []
    for filename in os.listdir(path):
        if filename.endswith('npy'):
            match = re.search(r'contact_pts_save_(\d+)\.npy$', filename)
            if match:
                number = int(match.group(1))
                print(f'Processing number: {number}')
                pts_path = os.path.join(path, f'contact_pts_save_{number}.npy')
                force_path = os.path.join(path, f'normal_forces_save_{number}.npy')
                contact_points = np.load(pts_path)
                normal_forces = np.load(force_path)
                filtered_pts = filter_pts(contact_points, normal_forces)
                print(filtered_pts.shape)
                pts_arrays.append(filtered_pts)
    pts_arrays = np.vstack(pts_arrays)
    print(pts_arrays.shape)
    np.save('./final_pts.npy', pts_arrays)
    visualize(pts_arrays, 'filtered_plot.png')

if __name__ == "__main__":
    # load_pretrained_weights()
    # load_normal_forces()
    # npy_path = "contact_pts_save_8.npy"
    # force_path = "normal_forces_save_8.npy"
    # load_contact_points(npy_path)
    # visualize(npy_path, force_path, 'contact_pts_save_8.png')
    combine_pts('./storage')
