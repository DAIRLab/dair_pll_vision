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

if __name__ == "__main__":
    load_pretrained_weights()