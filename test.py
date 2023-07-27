import torch
import sys
torch.set_printoptions(threshold=sys.maxsize)
traj_file_path = './assets/bundlesdf/0.pt'
traj = torch.load(traj_file_path)
print(traj)