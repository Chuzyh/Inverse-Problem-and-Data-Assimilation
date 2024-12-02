import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from glob import glob  # 用于加载多个数据集文件

# Define the MLP model
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

# Load all datasets
save_dir = "./data"  # 数据集保存的目录
ensemble_ml_files = glob(os.path.join(save_dir, "all_ensemble_ml_*.npy"))
ensemble_forecast_files = glob(os.path.join(save_dir, "all_ensemble_forecast_*.npy"))

# Initialize lists to store data
all_X = []
all_y = []

# Combine data from all datasets
for ml_file, forecast_file in zip(ensemble_ml_files, ensemble_forecast_files):
    all_ensemble_ml = np.load(ml_file)
    all_ensemble_forecast = np.load(forecast_file)

    # Reshape data for training
    time_steps, ensemble_size, state_size = all_ensemble_ml.shape
    X = all_ensemble_forecast.reshape(-1, state_size)  # Input features (forecast)
    y = all_ensemble_ml.reshape(-1, state_size)       # Target labels (analysis)

    all_X.append(X)
    all_y.append(y)

# Concatenate all data
all_X = np.vstack(all_X)
all_y = np.vstack(all_y)

# Convert to PyTorch tensors
X_tensor = torch.tensor(all_X, dtype=torch.float32)
y_tensor = torch.tensor(all_y, dtype=torch.float32)

# Shuffle the data
shuffle_indices = torch.randperm(len(X_tensor))
X_tensor = X_tensor[shuffle_indices]
y_tensor = y_tensor[shuffle_indices]

# Split data into training and validation sets
split_ratio = 0.8
split_idx = int(len(X_tensor) * split_ratio)
X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]

# Hyperparameters
input_dim = state_size
hidden_dim = 80  # Number of hidden units
output_dim = state_size
batch_size = 128

# Create data loaders
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function
model = EnsembleMLModel(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()

# Training in stages with decreasing learning rates
learning_rates = [0.00001]  # From high to low
num_epochs_per_stage = [100]       # Epochs for each stage
pretrained_model_path = os.path.join(save_dir, "model_80_1500_7.pth")
if os.path.exists(pretrained_model_path):
    print(f"Loading pretrained model from {pretrained_model_path}")
    model.load_state_dict(torch.load(pretrained_model_path))
else:
    print("No pretrained model found. Starting from scratch.")
for stage, (lr, epochs) in enumerate(zip(learning_rates, num_epochs_per_stage)):
    print(f"\nTraining Stage {stage + 1}: Learning Rate = {lr}, Epochs = {epochs}")
    
    # Set optimizer with current learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop for the current stage
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{epochs} (Stage {stage + 1}), Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

# Save the trained model
model_save_path = os.path.join(save_dir, "model_80_1500_8.pth")
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Evaluate on the validation set
model.eval()
with torch.no_grad():
    predictions = model(X_val)
    val_rmse = torch.sqrt(((predictions - y_val) ** 2).mean()).item()
    print(f"\nFinal Validation RMSE: {val_rmse:.6f}")
