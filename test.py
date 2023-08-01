import torch
import sys
import pickle
torch.set_printoptions(threshold=sys.maxsize)

if __name__ == "__main__":
    # traj_file_path = './assets/bundlesdf/0.pt'
    # traj = torch.load(traj_file_path)
    # print(traj)
    
    
    file_path = './examples/storage/bundlesdf/runs/bundlesdf/statistics.pkl'
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        print(data)