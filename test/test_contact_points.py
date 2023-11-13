import numpy as np

import os 

curr_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = "../contact_points.npy"
file_path = os.path.join(curr_dir, relative_path)
contact_points_array = np.load(file_path)
print(contact_points_array)