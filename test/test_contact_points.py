import numpy as np
import torch
import os 
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helpers.train_icnn import visualize_dirs_and_pts

FORCE_THRES = 0.3676 #N

def load_contact_points(path):
    """
    Contact points has shape (N, batch_size, 3)
    """
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = f"../{path}"
    file_path = os.path.join(curr_dir, relative_path)
    contact_points = torch.load(file_path)
    print(contact_points.shape)
    return contact_points.detach().numpy()

def load_directions(path):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = f"../{path}"
    file_path = os.path.join(curr_dir, relative_path)
    directions = torch.load(file_path)
    print(directions.shape)
    return directions.detach().numpy()

def load_forces(path):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = f"../{path}"
    file_path = os.path.join(curr_dir, relative_path)
    directions = torch.load(file_path)
    print(directions.shape)
    return directions.detach().numpy()

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

def filter_pts_and_dirs(contact_points, directions, normal_forces):
    '''
    Filter out points that are not in contact with the ground.
    '''
    assert normal_forces.ndim == 1
    assert contact_points.ndim == directions.ndim == 2
    assert normal_forces.shape[0] == contact_points.shape[0] == directions.shape[0]
    assert contact_points.shape[1] == directions.shape[1] == 3
    mask = normal_forces > FORCE_THRES
    filtered_points = contact_points[mask]
    filtered_directions = directions[mask]
    # flattened_filtered_points = filtered_points.reshape(-1, 3)

    print(f'{points.shape=}')
    print(f'{filtered_points.shape=}')

    visualize_and_filter(directions, contact_points, filtered_directions, filtered_points)
    # return flattened_filtered_points

def visualize_and_filter(directions, points, filtered_directions, filtered_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=10, label='unfiltered')
    ax.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2], s=30, label='filtered')
    # origin = [0, 0, 0]
    # for i in range(len(directions)):
    #     ax.quiver(*origin, *directions[i], length=np.linalg.norm(points[i]), arrow_length_ratio=0.1, color='r')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.legend()
    plt.show()
    # plt.savefig(img_path)

def visualize_forces(normal_forces, img_path=None):
    fig = plt.figure()
    plt.hist(normal_forces.flatten(), bins=30)
    plt.xlabel('contact impulse (Ns)')
    plt.show()

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

if __name__ == "__main__":
    # load_pretrained_weights()
    # combine_pts('./storage')
    points = load_contact_points('points.pt')
    directions = load_directions('directions.pt')
    forces = load_forces('normal_forces.pt')
    # visualize_forces(forces)
    filter_pts_and_dirs(points,directions,forces.flatten()/0.0068)