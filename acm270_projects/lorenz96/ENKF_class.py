from numerical_model import lorenz96,lorenz96_twoscale
import numpy as np
import torch
import torch.nn as nn
def rk4_step(func, t, x, dt, *args):
    """Perform a single 4th-order Runge–Kutta step."""
    k1 = func(t, x, *args)
    k2 = func(t + 0.5 * dt, x + 0.5 * dt * k1, *args)
    k3 = func(t + 0.5 * dt, x + 0.5 * dt * k2, *args)
    k4 = func(t + dt, x + dt * k3, *args)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
def generate_truth(N=40, n_subgrid=5, dt=0.005, total_time=5, F=8):
    """Generate the true trajectory and observations using the two-scale Lorenz96 model."""
    eps=1e-8
    t_eval = np.arange(0, total_time+eps, dt)
    print(t_eval)
    x0 = np.random.randn(N)  # Initialize main variables
    y0 = np.random.randn(n_subgrid, N)  # Initialize subgrid variables
    x0_twoscale = np.hstack([x0, y0.flatten()])

    trajectory = [x0_twoscale]
    for t in t_eval[:-1]:
        trajectory.append(rk4_step(lorenz96_twoscale, t, trajectory[-1], dt, N, n_subgrid, F))
    trajectory = np.array(trajectory)
    truth_trajectory = trajectory[:, :N]  # Extract the main variables
    return t_eval, truth_trajectory

def generate_observations(truth_trajectory, observation_error_std=0.5, obs_interval=0.05,dt=0.005):
    """Generate noisy observations at regular intervals."""
    t_obs = np.arange(0, truth_trajectory.shape[0] * dt, obs_interval)
    obs_indices = np.round(t_obs / dt).astype(int)
    observations = truth_trajectory[obs_indices] + np.random.normal(
        scale=observation_error_std, size=(len(obs_indices), truth_trajectory.shape[1])
    )
    return obs_indices, observations

# Define the EnKF class with Localization
class EnKF:
    def __init__(self, ensemble_size, observation_noise_std=0.5,inflation_factor=1.5):
        self.ensemble_size = ensemble_size
        self.observation_noise_std = observation_noise_std
        self.inflation_factor = inflation_factor

    def update(self, ensemble, observation,observation_error_std=0.5,dt=0.05,N=40):
        ensemble_forecast = np.zeros_like(ensemble)
        for i in range(self.ensemble_size):
            for _ in range(int(dt / 0.005)):  # Sub-step with finer resolution
                ensemble[i] = rk4_step(lorenz96, 0, ensemble[i], 0.005)
            ensemble_forecast[i] = ensemble[i]
        ensemble_forecast_mean = ensemble_forecast.mean(axis=0)
        ensemble_forecast = (
            ensemble_forecast_mean + self.inflation_factor * (ensemble_forecast - ensemble_forecast_mean)
        )
        Pf = np.cov(ensemble_forecast.T)
        H = np.eye(N)
        R = np.eye(H.shape[0]) * observation_error_std**2
        K = Pf @ H.T @ np.linalg.inv(H @ Pf @ H.T + R)

        # Update step
        y = observation
        y_expanded = np.tile(y, (self.ensemble_size, 1)).T
        ensemble_analysis = ensemble_forecast + (K @ (y_expanded - H @ ensemble_forecast.T)).T

        ensemble = ensemble_analysis
        return ensemble,ensemble_forecast
    
class EnKF_2ensemble:
    def __init__(self, ensemble_size, observation_noise_std=0.5,inflation_factor=1.5):
        self.ensemble_size = ensemble_size
        self.observation_noise_std = observation_noise_std
        self.inflation_factor = inflation_factor

    def update(self, ensemble1,ensemble2, observation,observation_error_std=0.5,dt=0.05,N=40):
        ensemble1_forecast = np.zeros_like(ensemble1)
        ensemble2_forecast = np.zeros_like(ensemble2)
        for i in range(self.ensemble_size//2):
            for _ in range(int(dt / 0.005)):  # Sub-step with finer resolution
                ensemble1[i] = rk4_step(lorenz96, 0, ensemble1[i], 0.005)
                ensemble2[i] = rk4_step(lorenz96, 0, ensemble2[i], 0.005)
            ensemble1_forecast[i] = ensemble1[i]
            ensemble2_forecast[i] = ensemble2[i]

        Pf = np.cov(ensemble1_forecast.T)
        H = np.eye(N)
        R = np.eye(H.shape[0]) * observation_error_std**2
        K = Pf @ H.T @ np.linalg.inv(H @ Pf @ H.T + R)

        # Update step
        y = observation
        y_expanded = np.tile(y, (self.ensemble_size//2, 1)).T
        
        ensemble2_analysis = ensemble2_forecast + (K @ (y_expanded - H @ ensemble2_forecast.T)).T
        Pf = np.cov(ensemble2_forecast.T)
        H = np.eye(N)
        R = np.eye(H.shape[0]) * observation_error_std**2
        K = Pf @ H.T @ np.linalg.inv(H @ Pf @ H.T + R)

        # Update step
        y = observation
        y_expanded = np.tile(y, (self.ensemble_size//2, 1)).T

        ensemble1_analysis = ensemble1_forecast + (K @ (y_expanded - H @ ensemble1_forecast.T)).T
        ensemble1 = ensemble1_analysis
        ensemble2 = ensemble2_analysis
        return ensemble1,ensemble2
class EnsembleMLModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EnsembleMLModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)
class EnKF_NN:
    def __init__(self, ensemble_size, observation_noise_std=0.5, inflation_factor=1.5, input_dim=40, hidden_dim=80, output_dim=40):
        self.ensemble_size = ensemble_size
        self.observation_noise_std = observation_noise_std
        self.inflation_factor = inflation_factor

        # 初始化 MLP 模型
        self.mlp = EnsembleMLModel(input_dim, hidden_dim, output_dim)

    def load_mlp_weights(self, model_path):
        """加载训练好的 MLP 模型参数"""
        self.mlp.load_state_dict(torch.load(model_path))
        self.mlp.eval()  # 设置为评估模式

    def update(self, ensemble, observation, observation_error_std=0.5, dt=0.05, N=40):
        ensemble_forecast = np.zeros_like(ensemble)
        for i in range(self.ensemble_size):
            for _ in range(int(dt / 0.005)):  # Sub-step with finer resolution
                ensemble[i] = rk4_step(lorenz96, 0, ensemble[i], 0.005)
            ensemble_forecast[i] = ensemble[i]
        
        ensemble_forecast_mean = ensemble_forecast.mean(axis=0)
    
        ensemble_forecast = (
            ensemble_forecast_mean + self.inflation_factor * (ensemble_forecast - ensemble_forecast_mean)
        )

        # 将预测值传递给 MLP
        ensemble_forecast_tensor = torch.tensor(ensemble_forecast, dtype=torch.float32)
        mlp_output = self.mlp(ensemble_forecast_tensor)

        # 使用 MLP 的输出作为新的预测
        ensemble_forecast = mlp_output.detach().numpy()  # 从 Tensor 转换回 numpy 数组
        
        # Kalman Gain and Update Step
        Pf = np.cov(ensemble_forecast.T)
        H = np.eye(N)
        R = np.eye(H.shape[0]) * observation_error_std**2
        K = Pf @ H.T @ np.linalg.inv(H @ Pf @ H.T + R)

        # Update step
        y = observation
        y_expanded = np.tile(y, (self.ensemble_size, 1)).T
        ensemble_analysis = ensemble_forecast + (K @ (y_expanded - H @ ensemble_forecast.T)).T

        ensemble = ensemble_analysis
        return ensemble, ensemble_forecast


