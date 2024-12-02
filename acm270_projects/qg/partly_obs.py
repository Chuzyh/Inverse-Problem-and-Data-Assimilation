from ml_model import model
from physical_model import drymodel
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
n_transient = 100  # 瞬态步长
n_steps = 40  # 总步数
ensemble_size = 10  # 集合大小
observation_interval = 1  # 观测间隔，每5步进行一次观测更新

# 随机初始状态
psi0 = np.random.randn(96, 192, 2)

# 数值模型模拟
psi = psi0
for i in range(n_transient):
    psi = drymodel(psi)
psi0=psi
# 集合初始化（随机扰动）
ensemble = np.random.randn(ensemble_size, *psi.shape) + psi

# 定义EnKF类
# 协方差矩阵局部化函数
def euclidean_distance(coord1, coord2):
    return np.sqrt(np.sum((coord1 - coord2) ** 2))
def compute_distance_matrix(observation_idx):
    n = len(observation_idx)
    distance_matrix = np.zeros((n, n))
    coords = np.array(np.unravel_index(observation_idx, (96,192, 2)))[:2]
    for i in range(n):
        for j in range(n):
            # 计算观测点之间的距离
            distance_matrix[i, j] = euclidean_distance(coords[:,i], coords[:,j])
    # print(distance_matrix)
    return distance_matrix

# 使用高斯函数进行局部化
def covariance_localization(perturbations, observation_idx, localization_radius):
    # 计算观测点之间的距离矩阵
    distance_matrix = compute_distance_matrix(observation_idx)
    
    # 计算高斯函数衰减因子
    # 高斯核的形式：exp(-distance^2 / (2 * localization_radius^2))
    localization_matrix = np.exp(-distance_matrix**2 / (2 * localization_radius**2))
    # print(perturbations @ perturbations.T)
    # 计算局部化的协方差矩阵
    localized_perturbations = perturbations @ perturbations.T #* localization_matrix
    return localized_perturbations

# 定义EnKF类，包含局部化协方差矩阵
class EnKF:
    def __init__(self, ensemble_size, observation_noise_std=0.1, localization_radius=30):
        self.ensemble_size = ensemble_size
        self.observation_noise_std = observation_noise_std
        self.localization_radius = localization_radius

    def update(self, ensemble, observation, observation_idx):
        # 只考虑观测间隔的状态变量
        for i in range(self.ensemble_size):
            ensemble[i]=drymodel(ensemble[i])
        H = np.zeros((observation.size, ensemble[0].size))  # 观测矩阵
        for i, idx in enumerate(observation_idx):
            H[i, idx] = 1.0  # 填充观测矩阵，只关注这些观测变量
        R = self.observation_noise_std**2 * np.eye(observation.size)  # 观测噪声协方差矩阵
        
        ensemble_mean = np.mean(ensemble, axis=0)  # 集合均值
        perturbations = ensemble - ensemble_mean  # 集合扰动

        # 计算观测的扰动
        obs_ensemble = H @ ensemble.reshape(self.ensemble_size, -1).T
        obs_ensemble_mean = np.mean(obs_ensemble, axis=1, keepdims=True)
        perturbed_observations = obs_ensemble - obs_ensemble_mean

        # 局部化协方差矩阵
        localized_perturbations = covariance_localization(perturbed_observations, observation_idx, self.localization_radius)
        
        # 计算卡尔曼增益
        kalman_gain = (perturbations.reshape(self.ensemble_size, -1).T @ perturbed_observations.T)
        kalman_gain = np.linalg.solve(localized_perturbations + R, kalman_gain.T).T
        print(kalman_gain)
        # 更新集合成员
        for i in range(self.ensemble_size):
            obs_perturbation = observation + np.random.normal(0, self.observation_noise_std, observation.shape)
            ensemble[i] += (kalman_gain @ (obs_perturbation - H @ ensemble[i].flatten())).reshape(ensemble[i].shape)

        return ensemble

# 生成物理和ML模型的预测
psi_phys = psi0
psis_phys = [psi0]
for i in range(n_steps):
    psi_phys = drymodel(psi_phys)
    psis_phys.append(psi_phys)

psi_ml = psi0
psis_ml = [psi0]
for i in range(n_steps):
    psi_ml = model.predict(psi_ml.transpose((1, 0, 2))[np.newaxis, :])[0, :, :, :].transpose((1, 0, 2))
    psis_ml.append(psi_ml)

# 生成合成观测（例如：每5步观测一次）
obs_dim=100
observation_idx = np.random.choice(psi0.size, size=obs_dim, replace=False)  # 随机选择50个观测点

print(observation_idx)
observations = np.array([psis_phys[i].flatten()[observation_idx] + np.random.normal(0, 0.1, obs_dim) for i in range(0, n_steps, observation_interval)])

# EnKF应用：每隔5步更新一次
ensemble_means = [np.mean(ensemble, axis=0)]
enkf = EnKF(ensemble_size)
for i in range(0, n_steps, observation_interval):
    observation = observations[i // observation_interval]  # 获取当前的观测
    ensemble = enkf.update(ensemble, observation, observation_idx)  # 更新集合
    ensemble_mean = np.mean(ensemble, axis=0)
    ensemble_means.append(ensemble_mean)


# 绘制物理模型与ML模型的RMSE
plt.plot(np.sqrt(np.mean((np.array(psis_ml) - np.array(psis_phys))**2, axis=(1, 2, 3))), label='ML forecast RMSE')
plt.plot(np.sqrt(np.mean((ensemble_means - np.array(psis_phys))**2, axis=(1, 2, 3))), label='EnKF filtered RMSE')
plt.legend()
plt.show()
