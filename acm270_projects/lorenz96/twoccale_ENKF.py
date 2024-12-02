import numpy as np
import matplotlib.pyplot as plt
from numerical_model import lorenz96,lorenz96_twoscale
eps=1e-8
# Define the single-scale Lorenz96 model
# RK4 integration step
def rk4_step(func, t, x, dt, *args):
    """Perform a single 4th-order Runge–Kutta step."""
    k1 = func(t, x, *args)
    k2 = func(t + 0.5 * dt, x + 0.5 * dt * k1, *args)
    k3 = func(t + 0.5 * dt, x + 0.5 * dt * k2, *args)
    k4 = func(t + dt, x + dt * k3, *args)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

# Generate truth trajectory and observations
def generate_truth(N=40, n_subgrid=8, dt=0.005, total_time=5, F=8):
    """Generate the true trajectory and observations using the two-scale Lorenz96 model."""
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

def generate_observations(truth_trajectory, observation_error_std=0.5, obs_interval=0.05):
    """Generate noisy observations at regular intervals."""
    t_obs = np.arange(0, truth_trajectory.shape[0] * dt, obs_interval)
    obs_indices = np.round(t_obs / dt).astype(int)
    observations = truth_trajectory[obs_indices] + np.random.normal(
        scale=observation_error_std, size=(len(obs_indices), truth_trajectory.shape[1])
    )
    return obs_indices, observations

# Ensemble Kalman Filter (EnKF) implementation
def enkf_single_scale(observations, obs_indices, H, R, x0, N_ens=20, inflation_factor=1.5, dt=0.05, F=8):
    """Ensemble Kalman Filter (EnKF) for the single-scale Lorenz96 model."""
    N = len(x0)  # State dimension
    n_obs = H.shape[0]  # Number of observed variables
    n_steps = len(obs_indices) - 1  # Number of assimilation steps

    # Initialize the ensemble
    ensemble = np.random.normal(x0, 0.5, size=(N_ens, N))

    # Store analysis states
    analysis_trajectory = np.zeros((n_steps + 1, N))
    analysis_trajectory[0] = x0

    for t_idx in range(n_steps):
        # Forecast step: Propagate each ensemble member forward
        ensemble_forecast = np.zeros_like(ensemble)
        for i in range(N_ens):
            for _ in range(int(dt / 0.005)):  # Sub-step with finer resolution
                ensemble[i] = rk4_step(lorenz96, 0, ensemble[i], 0.005, F)
            ensemble_forecast[i] = ensemble[i]

        # Inflation
        ensemble_forecast_mean = ensemble_forecast.mean(axis=0)
        ensemble_forecast = (
            ensemble_forecast_mean + inflation_factor * (ensemble_forecast - ensemble_forecast_mean)
        )

        # Compute Kalman gain
        Pf = np.cov(ensemble_forecast.T)
        K = Pf @ H.T @ np.linalg.inv(H @ Pf @ H.T + R)

        # Update step
        y = observations[t_idx + 1]
        y_expanded = np.tile(y, (N_ens, 1)).T
        ensemble_analysis = ensemble_forecast + (K @ (y_expanded - H @ ensemble_forecast.T)).T

        # Store analysis mean
        analysis_trajectory[t_idx + 1] = ensemble_analysis.mean(axis=0)
        ensemble = ensemble_analysis

    return analysis_trajectory

# Main execution
N = 40  # State size
n_subgrid = 8  # Subgrid variables for two-scale model
dt = 0.005
obs_interval = 0.05
observation_error_std = 0.5
total_time = 5

# Generate truth trajectory and observations
t_eval, truth_trajectory = generate_truth(N=N, n_subgrid=n_subgrid, dt=dt, total_time=total_time)
obs_indices, observations = generate_observations(
    truth_trajectory, observation_error_std=observation_error_std, obs_interval=obs_interval
)

# Observation operator and error covariance
print(obs_indices)
H = np.eye(N)  # Observe every other variable
R = np.eye(H.shape[0]) * observation_error_std**2

# Run EnKF
x0 = observations[0]  # Initial state
analysis_trajectory = enkf_single_scale(
    observations,
    obs_indices,
    H,
    R,
    x0,
    N_ens=200,
    inflation_factor=1.5,
    dt=0.05,
    F=8,
)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(t_eval, truth_trajectory[:, 0], label="True Trajectory (Variable 0)", color="blue")
plt.plot(t_eval[obs_indices], analysis_trajectory[:, 0], label="EnKF Analysis (Variable 0)", color="orange")
plt.scatter(t_eval[obs_indices], observations[:, 0], label="Observations (Variable 0)", color="red", s=10)
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.title("True Trajectory vs EnKF Analysis")
plt.show()
from sklearn.metrics import mean_squared_error
import numpy as np

# Function to compute RMSE
def rmse(true, forecast):
    return np.sqrt(mean_squared_error(true, forecast))

# Function to compute MAE
def mae(true, forecast):
    return np.mean(np.abs(true - forecast))

# Function to compute Correlation Coefficient
def correlation_coefficient(true, forecast):
    return np.corrcoef(true, forecast)[0, 1]

# Ensure that the observation indices and truth indices are aligned
observed_trajectory = truth_trajectory[obs_indices]  # Extract truth at observation times

# # Calculate RMSE, MAE, and Correlation for Observations vs True Trajectory
# obs_rmse = rmse(observed_trajectory[:, 0], observations[:, 0])
# obs_mae = mae(observed_trajectory[:, 0], observations[:, 0])
# obs_cc = correlation_coefficient(observed_trajectory[:, 0], observations[:, 0])

# # Calculate RMSE, MAE, and Correlation for EnKF Analysis vs True Trajectory
# enkf_rmse = rmse(observed_trajectory[:, 0], analysis_trajectory[:, 0])
# enkf_mae = mae(observed_trajectory[:, 0], analysis_trajectory[:, 0])
# enkf_cc = correlation_coefficient(observed_trajectory[:, 0], analysis_trajectory[:, 0])

# # Output the results
# print(f"Observation vs True:")
# print(f"RMSE: {obs_rmse:.4f}, MAE: {obs_mae:.4f}, Correlation Coefficient: {obs_cc:.4f}\n")

# print(f"EnKF Analysis vs True:")
# print(f"RMSE: {enkf_rmse:.4f}, MAE: {enkf_mae:.4f}, Correlation Coefficient: {enkf_cc:.4f}")


plt.figure(figsize=(10, 6))
plt.plot(np.sqrt(((observations - truth_trajectory[obs_indices])**2).mean(axis=1)), label='Obs RMSE')
plt.plot(np.sqrt(((analysis_trajectory - truth_trajectory[obs_indices])**2).mean(axis=1)), label='EnKF filtered RMSE')
plt.title("RMSE between ML forecast and Physical model with and without EnKF")
plt.xlabel("Time step")
plt.ylabel("RMSE")
plt.legend()
plt.savefig("rmse_with_filter.pdf")
plt.show()

# Plot the true trajectory, the ML forecast, and the EnKF-filtered forecast
plt.figure(figsize=(10, 6))
plt.plot(truth_trajectory[obs_indices, 0], label='True trajectory', linestyle='--')
plt.plot(observations[:, 0], label='Obs', linestyle='-.')
plt.plot(analysis_trajectory[:,0], label='EnKF forecast', linestyle='-')
plt.title("True trajectory vs ML forecast vs EnKF forecast")
plt.xlabel("Time step")
plt.ylabel("State variable")
plt.legend()
plt.savefig("forecast_with_filter.pdf")
plt.show()
