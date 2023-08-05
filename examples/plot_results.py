from collections import defaultdict
import sys

import json
import math
import os
import os.path as op
import pdb  # noqa
import re
from typing import Any, DefaultDict, List, Tuple

from matplotlib import rc, rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullFormatter
import numpy as np
import pickle

STATS_FILE = os.path.join(os.path.dirname(__file__), '..', 'results', 'storage', 'runs', 'bundlesdf', 'statistics.pkl')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'storage')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots')

keys_to_plot = {
    'train_model_loss_mean': '',
    'train_model_trajectory_mse_mean': '',
    'train_model_pos_int_traj_mean': '',
    'train_model_angle_int_traj_mean': '',
    'test_model_trajectory_mse_mean': '',
    'test_model_pos_int_traj_mean': '',
    'test_model_angle_int_traj_mean': '',
    'test_model_loss_mean': '',
}
with open(STATS_FILE, 'rb') as f:
    data = pickle.load(f)
    keys = []
    for key in data.keys():
        if key in keys_to_plot.keys():
            print(data[key])