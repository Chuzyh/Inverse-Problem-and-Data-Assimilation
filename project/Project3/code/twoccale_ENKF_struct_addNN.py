from ml_model import nn
import numpy as np
from ENKF_class import generate_truth,generate_observations,EnKF_NN,EnKF
import matplotlib.pyplot as plt

def compute_crps(ensemble, true_state):
    """
    计算高维 CRPS 值.
    Args:
        ensemble: 预测集合, shape (M, 96, 192, 2).
        true_state: 真实状态, shape (96, 192, 2).
    Returns:
        CRPS 值 (float).
    """
    M = ensemble.shape[0]
    ensemble_flat = ensemble.reshape(M, -1)  
    true_state_flat = true_state.flatten()  

    part1 = np.mean(np.linalg.norm(ensemble_flat - true_state_flat, ord=2, axis=1))
    pairwise_diff = ensemble_flat[:, np.newaxis, :] - ensemble_flat[np.newaxis, :, :]
    part2 = np.mean(np.linalg.norm(pairwise_diff, ord=2, axis=2))
    crps = part1 - part2/2

    return crps

n_subgrid = 5
N = 40  # State size
dt = 0.005
obs_interval = 0.05
observation_error_std = 0.5
total_time = 5
n_steps=100
ensemble_size = 100

t_eval, truth_trajectory = generate_truth(N=N, n_subgrid=n_subgrid, dt=dt, total_time=total_time)
obs_indices, observations = generate_observations(
    truth_trajectory, observation_error_std=observation_error_std, obs_interval=obs_interval
)
# Initialize the EnKF
enkf_nn = EnKF_NN(ensemble_size)
enkf_nn.load_mlp_weights("/mnt/c/workspace/study/Inverse Problem and Data Assimilation/acm270_projects/lorenz96/data/model_80_1500_4.pth")
# Initial ensemble of predictions for the ML model (this is just a random perturbation of the initial condition)
ensemble_nn = np.random.normal(observations[0], observation_error_std, size=(ensemble_size, N))
ensemble_nn_mean = np.zeros((n_steps+1,40))
ensemble_nn_mean[0] = np.mean(ensemble_nn, axis=0)
crps_nn=[compute_crps(ensemble_nn,truth_trajectory[obs_indices][0])]
# Run the model with EnKF and apply updates
for i in range(1,n_steps+1):
    # Perform the EnKF update using the synthetic observation and the ML prediction
    ensemble_nn,*rest= enkf_nn.update(ensemble_nn, observations[i])
    ensemble_nn_mean[i]=np.mean(ensemble_nn, axis=0)
    crps_nn.append(compute_crps(ensemble_nn,truth_trajectory[obs_indices][i]))


enkf = EnKF(ensemble_size)
# Initial ensemble of predictions for the ML model (this is just a random perturbation of the initial condition)
ensemble = np.random.normal(observations[0], observation_error_std, size=(ensemble_size, N))
ensemble_mean = np.zeros((n_steps+1,40))
ensemble_mean[0] = np.mean(ensemble, axis=0)
crps=[compute_crps(ensemble,truth_trajectory[obs_indices][0])]
# Run the model with EnKF and apply updates
for i in range(1,n_steps+1):
    # Perform the EnKF update using the synthetic observation and the ML prediction
    ensemble,*rest= enkf.update(ensemble, observations[i])
    ensemble_mean[i]=np.mean(ensemble, axis=0)
    crps.append(compute_crps(ensemble,truth_trajectory[obs_indices][i]))
obs_crps=[]
for obs, true_state in zip(observations, truth_trajectory[obs_indices]):
    obs_crps.append(compute_crps(np.array([obs]),true_state))
obs_crps=np.array(obs_crps)

# 创建一个包含三个子图的画布
fig, axes = plt.subplots(3, 1, figsize=(10, 18))
plt.subplots_adjust(hspace=0.3)  # 调整子图之间的间距

# 子图 1: RMSE Plot
axes[0].plot(np.sqrt(((observations - truth_trajectory[obs_indices])**2).mean(axis=1)), label='Obs RMSE')
axes[0].plot(np.sqrt(((ensemble_nn_mean - truth_trajectory[obs_indices])**2).mean(axis=1)), label='EnKF with NN filtered RMSE')
axes[0].plot(np.sqrt(((ensemble_mean - truth_trajectory[obs_indices])**2).mean(axis=1)), label='EnKF filtered RMSE')
axes[0].set_title("RMSE Plot")
axes[0].set_xlabel("Time Step")
axes[0].set_ylabel("RMSE")
axes[0].legend()

# 子图 2: Trajectory Plot
axes[1].plot(truth_trajectory[obs_indices, 0], label='True Trajectory', linestyle='--')
axes[1].plot(observations[:, 0], label='Obs', linestyle='-.')
axes[1].plot(ensemble_nn_mean[:, 0], label='EnKF with NN Forecast', linestyle='-')
axes[1].plot(ensemble_mean[:, 0], label='EnKF Forecast', linestyle=':')
axes[1].set_title("Trajectory Plot")
axes[1].set_xlabel("Time Step")
axes[1].set_ylabel("State Variable")
axes[1].legend()

# 子图 3: CRPS Comparison
axes[2].plot(crps, label='EnKF CRPS', color='blue')
axes[2].plot(crps_nn, label='EnKF with NN CRPS', color='green')
axes[2].plot(obs_crps, label='Observation CRPS', color='red')
axes[2].set_title("CRPS Comparison")
axes[2].set_xlabel("Time Step")
axes[2].set_ylabel("CRPS")
axes[2].legend()

# 保存整张图像
plt.savefig("combined_plots.pdf")

# 显示整张图像
plt.show()

from sklearn.metrics import mean_squared_error
def rmse(true, forecast):
    return np.sqrt(mean_squared_error(true, forecast))

# Function to compute MAE
def mae(true, forecast):
    return np.mean(np.abs(true - forecast))

# Ensure that the observation indices and truth indices are aligned
observed_trajectory = truth_trajectory[obs_indices]  # Extract truth at observation times

# Calculate RMSE, MAE, and Correlation for Observations vs True Trajectory
obs_rmse = rmse(observed_trajectory[:, 0], observations[:, 0])
obs_mae = mae(observed_trajectory[:, 0], observations[:, 0])
obs_cc = np.mean(obs_crps)

# Calculate RMSE, MAE, and Correlation for EnKF Analysis vs True Trajectory
enkf_rmse = rmse(observed_trajectory[:, 0], ensemble_mean[:, 0])
enkf_mae = mae(observed_trajectory[:, 0], ensemble_mean[:, 0])
enkf_cc = np.mean(crps)

# Calculate RMSE, MAE, and Correlation for EnKF Analysis vs True Trajectory
enkf_nn_rmse = rmse(observed_trajectory[:, 0], ensemble_nn_mean[:, 0])
enkf_nn_mae = mae(observed_trajectory[:, 0], ensemble_nn_mean[:, 0])
enkf_nn_cc = np.mean(crps_nn)

# Output the results
print(f"Observation vs True:")
print(f"RMSE: {obs_rmse:.4f}, MAE: {obs_mae:.4f}, Mean CRPS: {obs_cc:.4f}\n")

print(f"EnKF Analysis vs True:")
print(f"RMSE: {enkf_rmse:.4f}, MAE: {enkf_mae:.4f}, Mean CRPS: {enkf_cc:.4f}")

print(f"EnKF with NN Analysis vs True:")
print(f"RMSE: {enkf_nn_rmse:.4f}, MAE: {enkf_nn_mae:.4f}, Mean CRPS: {enkf_nn_cc:.4f}")

