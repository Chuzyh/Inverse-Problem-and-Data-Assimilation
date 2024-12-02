from ml_model import nn
import numpy as np
from ENKF_class import generate_truth,generate_observations,EnKF,EnKF_2ensemble
import matplotlib.pyplot as plt

n_subgrid = 8 
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
enkf = EnKF_2ensemble(ensemble_size)

# Initial ensemble of predictions for the ML model (this is just a random perturbation of the initial condition)
ensemble_ml1 = np.random.normal(observations[0], 0.5, size=(ensemble_size//2, N))
ensemble_ml2 = np.random.normal(observations[0], 0.5, size=(ensemble_size//2, N))
ensemble_ml_mean1 = np.zeros((n_steps+1,40))
ensemble_ml_mean2 = np.zeros((n_steps+1,40))
ensemble_ml_mean1[0] = np.mean(ensemble_ml1, axis=0)
ensemble_ml_mean2[0] = np.mean(ensemble_ml2, axis=0)
# Run the model with EnKF and apply updates
for i in range(1,n_steps+1):
    # Perform the EnKF update using the synthetic observation and the ML prediction
    ensemble_ml1,ensemble_ml2,*rest= enkf.update(ensemble_ml1,ensemble_ml2, observations[i])
    ensemble_ml_mean1[i]=np.mean(ensemble_ml1, axis=0)
    ensemble_ml_mean2[i]=np.mean(ensemble_ml2, axis=0)
ensemble_ml_mean_2ensemble=(ensemble_ml_mean1+ensemble_ml_mean2)/2


enkf2 = EnKF(ensemble_size)
# Initial ensemble of predictions for the ML model (this is just a random perturbation of the initial condition)
ensemble_ml = np.random.normal(observations[0], 0.5, size=(ensemble_size, N))
ensemble_ml_mean = np.zeros((n_steps+1,40))
ensemble_ml_mean[0] = np.mean(ensemble_ml, axis=0)
# Run the model with EnKF and apply updates
for i in range(1,n_steps+1):
    # Perform the EnKF update using the synthetic observation and the ML prediction
    ensemble_ml,*rest= enkf2.update(ensemble_ml, observations[i])
    ensemble_ml_mean[i]=np.mean(ensemble_ml, axis=0)

enkf3 = EnKF(ensemble_size,inflation_factor=1)
# Initial ensemble of predictions for the ML model (this is just a random perturbation of the initial condition)
ensemble_ml = np.random.normal(observations[0], 0.5, size=(ensemble_size, N))
ensemble_ml_mean_noinflation = np.zeros((n_steps+1,40))
ensemble_ml_mean_noinflation[0] = np.mean(ensemble_ml, axis=0)
# Run the model with EnKF and apply updates
for i in range(1,n_steps+1):
    # Perform the EnKF update using the synthetic observation and the ML prediction
    ensemble_ml,*rest= enkf3.update(ensemble_ml, observations[i])
    ensemble_ml_mean_noinflation[i]=np.mean(ensemble_ml, axis=0)
# Plot the RMSE between the physical and ML forecast
plt.figure(figsize=(10, 6))
plt.plot(np.sqrt(((observations - truth_trajectory[obs_indices])**2).mean(axis=1)), label='Obs RMSE')
plt.plot(np.sqrt(((ensemble_ml_mean - truth_trajectory[obs_indices])**2).mean(axis=1)), label='EnKF filtered RMSE')
plt.plot(np.sqrt(((ensemble_ml_mean_noinflation - truth_trajectory[obs_indices])**2).mean(axis=1)), label='EnKF filtered RMSE no inflation')
plt.plot(np.sqrt(((ensemble_ml_mean_2ensemble - truth_trajectory[obs_indices])**2).mean(axis=1)), label='EnKF with two ensembles filtered RMSE')
# plt.title("RMSE between ML forecast and Physical model with and without EnKF")
plt.xlabel("Time step")
plt.ylabel("RMSE")
plt.legend()
plt.savefig("rmse_with_filter.pdf")
plt.show()

# Plot the true trajectory, the ML forecast, and the EnKF-filtered forecast
plt.figure(figsize=(10, 6))
plt.plot(truth_trajectory[obs_indices, 0], label='True trajectory', linestyle='--')
plt.plot(observations[:, 0], label='Obs', linestyle='-.')
plt.plot(ensemble_ml_mean[:,0], label='EnKF forecast', linestyle='-')
plt.plot(ensemble_ml_mean_2ensemble[:,0], label='EnKF two ensemble forecast', linestyle=':')
plt.plot(ensemble_ml_mean_noinflation[:,0], label='EnKF forecast no inflation', linestyle=':')
# plt.title("True trajectory vs ML forecast vs EnKF forecast")
plt.xlabel("Time step")
plt.ylabel("State variable")
plt.legend()
plt.savefig("forecast_with_filter.pdf")
plt.show()


from sklearn.metrics import mean_squared_error
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

# Calculate RMSE, MAE, and Correlation for Observations vs True Trajectory
obs_rmse = rmse(observed_trajectory[:, 0], observations[:, 0])
obs_mae = mae(observed_trajectory[:, 0], observations[:, 0])
obs_cc = correlation_coefficient(observed_trajectory[:, 0], observations[:, 0])

# Calculate RMSE, MAE, and Correlation for EnKF Analysis vs True Trajectory
enkf_rmse = rmse(observed_trajectory[:, 0], ensemble_ml_mean[:, 0])
enkf_mae = mae(observed_trajectory[:, 0], ensemble_ml_mean[:, 0])
enkf_cc = correlation_coefficient(observed_trajectory[:, 0], ensemble_ml_mean[:, 0])

# Output the results
print(f"Observation vs True:")
print(f"RMSE: {obs_rmse:.4f}, MAE: {obs_mae:.4f}, Correlation Coefficient: {obs_cc:.4f}\n")

print(f"EnKF Analysis vs True:")
print(f"RMSE: {enkf_rmse:.4f}, MAE: {enkf_mae:.4f}, Correlation Coefficient: {enkf_cc:.4f}")

