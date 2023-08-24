from dair_pll import file_utils
from dair_pll.dataset_management import ExperimentDataManager
from dair_pll.deep_learnable_system import DeepLearnableSystemConfig
from dair_pll.drake_experiment import DrakeDeepLearnableExperiment, DrakeMultibodyLearnableExperiment, MultibodyLearnableSystemConfig
from dair_pll.experiment import TrainingState
import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import os
from scipy.spatial.transform import Rotation as R

def load_pkl(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    print(data.keys())
    
def visualize_trajectory(trajectory_dir, fig_name):
    traj = torch.load(trajectory_dir) #q_t(wxyz), p_t, w_t, dp_t
    print(f'traj loaded: {traj.size()}') #N,13
    p_t = traj[:,4:7].numpy() #N,3
    q_t = traj[:,:4].numpy() #N,4, w,x,y,z
    q_t_shuffled = np.concatenate((q_t[:, 1:], q_t[:, 0].reshape(-1,1)), axis=1) ##N,4, x,y,z,w
    dp_t = traj[:,10:].numpy() #N,3
    w_t_body = traj[:,7:10].numpy() #N,3, in body frame
    fig, ax = plt.subplots(4, 3, figsize=(15, 15))

    # Plot positions
    ax[0, 0].plot(p_t[:, 0])
    ax[0, 0].set_title('X Position')
    ax[0, 1].plot(p_t[:, 1])
    ax[0, 1].set_title('Y Position')
    ax[0, 2].plot(p_t[:, 2])
    ax[0, 2].set_title('Z Position')

    # Plot Quaternion components
    ax[1, 0].plot(q_t[:, 0])
    ax[1, 0].set_title('Quaternion w')
    ax[1, 1].plot(q_t[:, 1])
    ax[1, 1].set_title('Quaternion x')
    ax[1, 2].plot(q_t[:, 2])
    ax[1, 2].set_title('Quaternion y')
    ax[2, 0].plot(q_t[:, 3])
    ax[2, 0].set_title('Quaternion z')

    # Plot angular velocities
    ax[2, 1].plot(w_t_body[:, 0])
    ax[2, 1].set_title('Angular Velocity X')
    ax[2, 2].plot(w_t_body[:, 1])
    ax[2, 2].set_title('Angular Velocity Y')
    ax[3, 0].plot(w_t_body[:, 2])
    ax[3, 0].set_title('Angular Velocity Z')

    # Plot linear velocities
    ax[3, 1].plot(dp_t[:, 0])
    ax[3, 1].set_title('Linear Velocity X')
    ax[3, 2].plot(dp_t[:, 1])
    ax[3, 2].set_title('Linear Velocity Y')

    plt.tight_layout()
    fig.suptitle('ContactNets cube trajectory')
    plt.savefig(fig_name)
    print(f'Saved to {fig_name}')
    plt.show()

    
def load_experiment(run_name):
    run_path = f"./results/{storage}/runs/{run_name}"
    storage_name = os.path.abspath(os.path.join(run_path, '..', '..'))
    print(storage_name)
    experiment_config = file_utils.load_configuration(storage_name, run_name)

    if isinstance(experiment_config.learnable_config,
                  MultibodyLearnableSystemConfig):
        experiment_config.learnable_config.randomize_initialization = False
        return DrakeMultibodyLearnableExperiment(experiment_config)
    elif isinstance(experiment_config.learnable_config,
                    DeepLearnableSystemConfig):
        return DrakeDeepLearnableExperiment(experiment_config)
    raise RuntimeError(f'Cannot recognize learnable type ' + \
                       f'{experiment_config.learnable_config}')
    
def get_test_set_traj_target_and_prediction(experiment):
    stats = file_utils.load_evaluation(experiment.config.storage,
                                       experiment.config.run_name)
    print("##########", len(stats['test_model_target_sample']))
    test_traj_target = stats['test_model_target_sample'][0]
    test_traj_prediction = stats['test_model_prediction_sample'][0]
    return Tensor(test_traj_target), Tensor(test_traj_prediction)
    
def get_best_system_from_experiment(exp):
    checkpoint_filename = file_utils.get_model_filename(exp.config.storage,
                                                        exp.config.run_name)
    checkpoint_dict = torch.load(checkpoint_filename)
    training_state = TrainingState(**checkpoint_dict)

    assert training_state.finished_training

    exp.learning_data_manager = ExperimentDataManager(
        exp.config.storage, exp.config.data_config,
        training_state.trajectory_set_split_indices)
    train_set, _, test_set = \
        exp.learning_data_manager.get_updated_trajectory_sets()
    learned_system = exp.get_learned_system(torch.cat(train_set.trajectories))
    learned_system.load_state_dict(training_state.best_learned_system_state)

    return learned_system

def load_experiment_run_dir_sys(storage, run_name):
    experiment = load_experiment(run_name)
    run_dir = f'./results/{storage}/runs/{run_name}'
    print(f'Loading {run_dir}')
    learned_system = get_best_system_from_experiment(experiment)
    return experiment, run_dir, learned_system

def split_traj(traj):
    p_t = traj[:,4:7].numpy() #N,3
    q_t = traj[:,:4].numpy() #N,4, w,x,y,z
    q_t_shuffled = np.concatenate((q_t[:, 1:], q_t[:, 0].reshape(-1,1)), axis=1) ##N,4, x,y,z,w
    dp_t = traj[:,10:].numpy() #N,3
    w_t_body = traj[:,7:10].numpy() #N,3, in body frame
    return p_t, q_t_shuffled, dp_t, w_t_body

def eval(storage, run_name):
    experiment, run_dir, learned_system = \
                load_experiment_run_dir_sys(storage, run_name)
    gt_traj, pred_traj = get_test_set_traj_target_and_prediction(
                experiment)
    print(gt_traj.size(), pred_traj.size())
    p_t_gt, q_t_gt, dp_t_gt, w_t_gt = split_traj(gt_traj)
    p_t_est, q_t_est, dp_t_est, w_t_est = split_traj(pred_traj)
    rot_gt = R.from_quat(q_t_gt).as_matrix()
    rot_est = R.from_quat(q_t_est).as_matrix()
    print(f'rot: {rot_gt.shape}, p_t: {p_t_gt.shape}')
    rot_err = np.linalg.norm(rot_gt - rot_est, 'fro', axis=(1,2))
    trans_err = np.linalg.norm(p_t_gt - p_t_est,axis=1)
    print(f'rot: {rot_err.shape}, trans: {trans_err.shape}')
    
    # max_len = max(len(traj) for traj in gt_trajs)
    timestamps = np.arange(rot_err.shape[0])
    # rot error
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, rot_err, '-o', markersize=4, lw=2)
    plt.title('Rotation Error (deg)')
    plt.xlabel('step')
    plt.ylabel('Rotation Error (deg)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'./results/{storage}/runs/{run}/{storage}_{run}_{toss_id}_rot.png')
    # trans error
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, trans_err, '-o', markersize=4, lw=2)
    plt.title('Translation Error (m)')
    plt.xlabel('step')
    plt.ylabel('Translation Error (m)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'./results/{storage}/runs/{run}/{storage}_{run}_{toss_id}_trans.png')
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--storage",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--run",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--toss_id",
        type=int,
        required=True,
    )
    args = parser.parse_args()
    storage = args.storage
    run = args.run
    toss_id = args.toss_id
    
    gt_traj = f'./results/{storage}/data/ground_truth/{toss_id}.pt'
    est_traj = f'./results/{storage}/data/learning/{toss_id}.pt'
    # visualize_trajectory(gt_traj, f'./results/{run}/gt_{run}_{toss_id}.png')
    # visualize_trajectory(est_traj, f'./results/{run}/est_{run}_{toss_id}.png')
    
    # stats = f'./results/{storage}/runs/{run}/statistics.pkl'
    # load_pkl(stats)
    
    eval(storage, run)