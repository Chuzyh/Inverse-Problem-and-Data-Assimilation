import os
from ENKF_class import generate_observations,generate_truth,EnKF
import numpy as np
np.random.seed(233)
def generate_data_with_different_initial_states(
    num_datasets=10,  # Number of datasets to generate
    save_dir="data",
    N=40, n_subgrid=5, dt=0.005, total_time=5, F=8, 
    obs_interval=0.05, 
    ensemble_size=80
):
    """Generate multiple datasets by varying initial states."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for dataset_id in range(num_datasets):
        
        # Generate true trajectory
        t_eval, truth_trajectory = generate_truth(N=N, n_subgrid=n_subgrid, dt=dt, total_time=total_time, F=F)
        if(np.any(np.isnan(truth_trajectory)) or np.any(np.isinf(truth_trajectory))):
            continue
        # Generate observations
        observation_error_std = 0.5#np.random.uniform(0.1, 0.8)
        inflation_factor = 1.5#np.random.uniform(1.2, 2)
        
        obs_indices, observations = generate_observations(
            truth_trajectory, observation_error_std=observation_error_std, obs_interval=obs_interval
        )
        
        # Initialize EnKF
        enkf = EnKF(ensemble_size, observation_noise_std=observation_error_std, inflation_factor=inflation_factor)
        ensemble_ml = np.random.normal(observations[0], 0.5, size=(ensemble_size, N))
        ensemble_ml_mean = np.zeros((len(obs_indices), N))
        ensemble_ml_mean[0] = np.mean(ensemble_ml, axis=0)
        
        all_ensemble_ml = []  # To store all ensemble_ml at each time step
        all_ensemble_forecast = []  # To store all ensemble_forecast at each time step
        # Run EnKF updates
        for i in range(1, len(obs_indices)):
            ensemble_ml,ensemble_forecast = enkf.update(ensemble_ml, observations[i])
            ensemble_ml_mean[i] = np.mean(ensemble_ml, axis=0)
            all_ensemble_ml.append(ensemble_ml.copy())  # Copy to avoid reference issues
            all_ensemble_forecast.append(ensemble_forecast.copy())
        # Save data
        all_ensemble_ml = np.array(all_ensemble_ml)  # Shape: (time_steps, ensemble_size, state_size)
        all_ensemble_forecast = np.array(all_ensemble_forecast)  # Shape: (time_steps, ensemble_size, state_size)
        if(np.any(np.isnan(all_ensemble_ml)) or np.any(np.isinf(all_ensemble_ml)) or np.any(np.isnan(all_ensemble_forecast)) or np.any(np.isinf(all_ensemble_forecast))):
            continue
        print(dataset_id,'ok')
        np.save(os.path.join(save_dir, f"all_ensemble_ml_{dataset_id}.npy"), all_ensemble_ml)
        np.save(os.path.join(save_dir, f"all_ensemble_forecast_{dataset_id}.npy"), all_ensemble_forecast)
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
        print(f"Dataset {dataset_id} saved.")

# Generate 10 datasets by varying initial states``
generate_data_with_different_initial_states(num_datasets=1500)
