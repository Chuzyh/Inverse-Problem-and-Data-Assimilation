import numpy as np
import matplotlib.pyplot as plt
from ml_model import model  # ML surrogate model
from physical_model import drymodel  # Numerical QG model

from concurrent.futures import ProcessPoolExecutor, as_completed
# Simulation parameters
n_transient = 100         # Number of transient steps to stabilize the initial state
n_steps = 40        # Number of time steps for trajectory
ensemble_size = 200        # Ensemble size for ML model
ensemble_size_phys = 20   # Ensemble size for physical model (smaller for higher computational cost)
assimilation_interval = 1.0  # Forecast interval (1 time unit ~ 6 hours)
observation_noise_std = 0.1  # Standard deviation of observation noise
localization_radius = 4
# Initialize the QG state
psi0 = np.random.randn(96, 192, 2)

# EnKF class to handle ensemble filtering
import numpy as np
from scipy.sparse import lil_matrix
def euclidean_distance(coord1, coord2):
    return np.sqrt(np.sum((coord1 - coord2) ** 2))

from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

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

# 全局变量用于缓存数据
global_perturbations_transpose = None

def initialize_global_variables(perturbations):
    """初始化全局变量，避免重复传递和计算"""
    global global_perturbations_transpose
    global_perturbations_transpose = perturbations.T

def compute_localized_covariance_for_index(i, localization_radius):
    """计算每个索引的局部协方差行，避免重复使用 perturbations"""
    global global_perturbations_transpose
    state_size = global_perturbations_transpose.shape[0]
    coords = np.array(np.unravel_index(i, (96, 192, 2)))[:2]
    x_min = max(0, coords[0] - localization_radius)
    x_max = min(coords[0] + localization_radius + 1, 96)
    y_min = max(0, coords[1] - localization_radius)
    y_max = min(coords[1] + localization_radius + 1, 192)
    
    # 仅生成当前行的稀疏矩阵
    row_result = lil_matrix((1, state_size))

    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            distance = euclidean_distance(coords, [x, y])
            if distance > localization_radius:
                continue
            localization_factor = max((localization_radius - distance) / localization_radius, 0)
            j0 = np.ravel_multi_index((x, y, 0), (96, 192, 2))
            j1 = np.ravel_multi_index((x, y, 1), (96, 192, 2))
            row_result[0, j0] = localization_factor * np.dot(global_perturbations_transpose[i], global_perturbations_transpose[j0])
            row_result[0, j1] = localization_factor * np.dot(global_perturbations_transpose[i], global_perturbations_transpose[j1])
    
    return i, row_result

def compute_localized_covariance(perturbations, localization_radius=localization_radius, max_processes=16):
    """主函数：计算局部协方差矩阵并初始化进程池中的全局变量"""
    state_size = perturbations.shape[1]
    sparse_cov_matrix = lil_matrix((state_size, state_size))

    # 初始化全局变量
    initialize_global_variables(perturbations)

    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        futures = [
            executor.submit(compute_localized_covariance_for_index, i, localization_radius)
            for i in range(state_size)
        ]

        for future in as_completed(futures):
            i, row_result = future.result()
            sparse_cov_matrix[i] = row_result

    return sparse_cov_matrix

from scipy.sparse.linalg import spsolve
class EnKF:
    def __init__(self, ensemble_size, observation_noise_std=observation_noise_std, inflation_factor=1.5):
        self.ensemble_size = ensemble_size
        self.observation_noise_std = observation_noise_std
        self.inflation_factor = inflation_factor

    def update(self, ensemble, observation, error_threshold=0.05):
        # Compute ensemble mean
        ensemble_mean = np.mean(ensemble, axis=0)
        perturbations = ensemble - ensemble_mean
        ensemble = ensemble_mean + self.inflation_factor * (ensemble - ensemble_mean)
        ensemble_mean = np.mean(ensemble, axis=0)
        perturbations = ensemble - ensemble_mean
        # 使用 float32 类型计算 Kalman 增益
        print("Start calc C")
        C=compute_localized_covariance(perturbations.reshape(perturbations.shape[0], -1))
        # C = self.inflation_factor * C
        # C = lil_matrix((perturbations.reshape(perturbations.shape[0], -1).shape[1], perturbations.reshape(perturbations.shape[0], -1).shape[1]))
        print(C.shape)
        R = lil_matrix(C.shape)
        for i in range(0,R.shape[0]):
            R[i,i]=observation_noise_std**2
            # C[i,i]=1
        A=C+R
        # Update ensemble members
        for i in range(self.ensemble_size):
            obs_perturbation = observation + np.random.normal(0, self.observation_noise_std, observation.shape)-ensemble[i]
            delta = spsolve(A, obs_perturbation.flatten())
            print(delta)
            ensemble[i] += (C*delta).reshape(ensemble[i].shape)
        rmse = np.sqrt(np.mean((ensemble - observation) ** 2))
        # if rmse > error_threshold:
        #     self.inflation_factor *= 1.1  # Increase inflation factor
        # else:
        #     self.inflation_factor *= 0.95 
        print("inflation_factor",self.inflation_factor)
        print("rmse",rmse)
        return ensemble


# Function to generate true trajectory and synthetic observations
def generate_true_trajectory_and_observations():
    psi = psi0
    for _ in range(n_transient):
        psi = drymodel(psi)  # Spin up the system

    # Generate true trajectory
    true_states = [psi]
    for _ in range(n_steps):
        psi = drymodel(psi)
        true_states.append(psi)
    
    # Generate synthetic observations with added Gaussian noise
    observations = [state + np.random.normal(0, observation_noise_std, state.shape) for state in true_states]
    return np.array(true_states), np.array(observations)

# Initialize the ensemble with slight perturbations
def initialize_ensemble(state, size):
    return np.array([state + np.random.normal(0, 0.1, state.shape) for _ in range(size)])

# Ensemble Kalman filter with ML model
def apply_enkf_ml(ensemble, observations,true_state):
    enkf = EnKF(ensemble_size)
    filtered_ensemble_ml = [ensemble.mean(axis=0)]
    crps=[compute_crps(ensemble,true_state[0])]
    for i in range(n_steps):
        # Forecast step using ML model
        ensemble = np.array([model.predict(e.transpose((1, 0, 2))[np.newaxis, :])[0, :, :, :].transpose((1, 0, 2)) for e in ensemble])
        
        # Update step using EnKF
        ensemble = enkf.update(ensemble, observations[i])
        filtered_ensemble_ml.append(ensemble.mean(axis=0))
        crps.append(compute_crps(ensemble,true_state[i]))
    
    return np.array(filtered_ensemble_ml),np.array(crps)

# Ensemble Kalman filter with numerical model
def apply_enkf_physical(ensemble, observations,true_state):
    enkf = EnKF(ensemble_size_phys)
    filtered_ensemble_phys = [ensemble.mean(axis=0)]
    crps=[compute_crps(ensemble,true_state[0])]
    for i in range(n_steps):
        # Forecast step using numerical model
        ensemble = np.array([drymodel(e) for e in ensemble])
        
        # Update step using EnKF
        ensemble = enkf.update(ensemble, observations[i])
        filtered_ensemble_phys.append(ensemble.mean(axis=0))
        crps.append(compute_crps(ensemble,true_state[i]))
    
    return np.array(filtered_ensemble_phys),np.array(crps)

# Compare filtering performance with RMSE
def compute_rmse(filtered_states, true_states):
    return np.sqrt(np.mean((filtered_states - true_states) ** 2, axis=(1, 2, 3)))

# Main script
true_states, observations = generate_true_trajectory_and_observations()
obs_crps=[]
for obs, true_state in zip(observations, true_states):
    obs_crps.append(compute_crps(np.array([obs]),true_state))
obs_crps=np.array(obs_crps)
print(true_states.shape)
print(observations.shape)

# Initialize ensembles
ensemble_ml = initialize_ensemble(observations[0], ensemble_size)
ensemble_phys = initialize_ensemble(observations[0], ensemble_size_phys)

# Apply EnKF with ML and numerical models
filtered_states_ml,ml_crps = apply_enkf_ml(ensemble_ml, observations,true_states)
filtered_states_phys,phys_crps = apply_enkf_physical(ensemble_phys, observations,true_states)

# # Calculate RMSE for each filtering method
rmse_obs = compute_rmse(observations, true_states)
rmse_ml = compute_rmse(filtered_states_ml, true_states)
rmse_phys = compute_rmse(filtered_states_phys, true_states)

# 创建两个子图
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 左侧图：RMSE 对比
axes[0].plot(rmse_ml, label='ML Model Filter RMSE', color='blue')
axes[0].plot(rmse_phys, label='Numerical Model Filter RMSE', color='green')
axes[0].plot(rmse_obs, label='Observation RMSE', color='red')
axes[0].set_xlabel('Time Step')
axes[0].set_ylabel('RMSE')
axes[0].legend()
axes[0].set_title('RMSE Comparison: ML vs Numerical Model')

# 右侧图：CRPS 对比
axes[1].plot(ml_crps, label='ML Model Filter CRPS', color='blue')
axes[1].plot(phys_crps, label='Numerical Model Filter CRPS', color='green')
axes[1].plot(obs_crps, label='Observation CRPS', color='red')
axes[1].set_xlabel('Time Step')
axes[1].set_ylabel('CRPS')
axes[1].legend()
axes[1].set_title('CRPS Comparison: ML vs Numerical Model')

# 调整布局
plt.tight_layout()
plt.show()
