from ml_model import nn
from numerical_model import lorenz96
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Time step for the ML model; use the same for the numerical integration
dt = 0.05
n_steps = 100

# Generate a random initial state
x0 = np.random.randn(40)

# Initialize arrays to store the state predictions
x_ml = np.zeros((n_steps, 40))
x_ml[0] = x0

# Define the EnKF class with Localization
class EnKF:
    def __init__(self, ensemble_size, observation_noise_std=0.5, localization_radius=5):
        self.ensemble_size = ensemble_size
        self.observation_noise_std = observation_noise_std
        self.localization_radius = localization_radius

    def localization_matrix(self, size):
        loc_matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                distance = abs(i - j)
                loc_matrix[i, j] = np.exp(-0.5 * (distance / self.localization_radius)**2)
        return loc_matrix

    def update(self, ensemble, observation):
        for i in range(self.ensemble_size):
            ensemble[i]= nn._smodel.predict(ensemble[i].reshape((1, 40, 1)))[0, :, 0]
        ensemble_mean = np.mean(ensemble, axis=0)
        perturbations = ensemble - ensemble_mean

        # Observation model: observation = state + noise
        H = np.eye(ensemble_mean.size)
        R = self.observation_noise_std**2 * np.eye(observation.size)

        # Calculate observation ensemble perturbations
        obs_ensemble = H @ ensemble.reshape(self.ensemble_size, -1).T
        obs_ensemble_mean = np.mean(obs_ensemble, axis=1, keepdims=True)
        perturbed_observations = obs_ensemble - obs_ensemble_mean

        # Compute Kalman gain and apply localization
        loc_matrix = self.localization_matrix(ensemble_mean.size)
        kalman_gain = (perturbations.reshape(self.ensemble_size, -1).T @ perturbed_observations.T* loc_matrix) @ \
                      np.linalg.inv(perturbed_observations @ perturbed_observations.T* loc_matrix + R)
        kalman_gain = kalman_gain 
        print(kalman_gain)
        # Update ensemble members
        for i in range(self.ensemble_size):
            obs_perturbation = observation + np.random.normal(0, self.observation_noise_std, observation.shape)
            ensemble[i] += (kalman_gain @ (obs_perturbation - H @ ensemble[i].flatten())).reshape(ensemble[i].shape)
        
        return ensemble

# Generate the true trajectory using the numerical model (Lorenz96)
x_phys = solve_ivp(lorenz96, [0, n_steps*dt], x0, t_eval=np.arange(0.0, n_steps*dt, dt)).y.T

# Generate synthetic observations by adding noise to the true trajectory
observation_noise_std = 0.3  # Standard deviation of observation noise
x_obs = x_phys + np.random.normal(0, observation_noise_std, x_phys.shape)

# Initialize the EnKF
ensemble_size = 20
enkf = EnKF(ensemble_size)

# Initial ensemble of predictions for the ML model (this is just a random perturbation of the initial condition)
ensemble_ml = np.random.randn(ensemble_size, 40) + x0
ensemble_ml_mean = np.zeros((n_steps,40))
ensemble_ml_mean[0] = np.mean(ensemble_ml, axis=0)
# Run the model with EnKF and apply updates
for i in range(1, n_steps):
    # Machine learning prediction step (this could be a neural network or any other model)
    x = nn._smodel.predict(x_ml[i-1].reshape((1, 40, 1)))[0, :, 0]
    x_ml[i] = x

    # Perform the EnKF update using the synthetic observation and the ML prediction
    ensemble_ml = enkf.update(ensemble_ml, x_obs[i])
    ensemble_ml_mean[i]=np.mean(ensemble_ml, axis=0)

# Plot the RMSE between the physical and ML forecast
plt.figure(figsize=(10, 6))
plt.plot(np.sqrt(((x_obs - x_phys)**2).mean(axis=1)), label='Obs RMSE')
plt.plot(np.sqrt(((ensemble_ml_mean - x_phys)**2).mean(axis=1)), label='EnKF filtered RMSE')
plt.title("RMSE between ML forecast and Physical model with and without EnKF")
plt.xlabel("Time step")
plt.ylabel("RMSE")
plt.legend()
plt.savefig("rmse_with_filter.pdf")
plt.show()

# Plot the true trajectory, the ML forecast, and the EnKF-filtered forecast
plt.figure(figsize=(10, 6))
plt.plot(x_phys[:, 0], label='True trajectory', linestyle='--')
plt.plot(x_obs[:, 0], label='Obs', linestyle='-.')
plt.plot(ensemble_ml_mean[:,0], label='EnKF forecast', linestyle='-')
plt.title("True trajectory vs ML forecast vs EnKF forecast")
plt.xlabel("Time step")
plt.ylabel("State variable")
plt.legend()
plt.savefig("forecast_with_filter.pdf")
plt.show()
