import numpy as np
import torch
import os 

def load_contact_points():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = "../contact_points.npy"
    file_path = os.path.join(curr_dir, relative_path)
    contact_points_array = np.load(file_path)
    print(contact_points_array)

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

if __name__ == "__main__":
    # load_pretrained_weights()
    load_contact_points()
    # obj_path = "/home/cnets-vision/mengti_ws/dair_pll_latest/results/cube_deep-13/runs/residual-4/urdfs/body.obj"
    # npy_path = "./contact_points.npy"
    # obj2npy(obj_path, npy_path)