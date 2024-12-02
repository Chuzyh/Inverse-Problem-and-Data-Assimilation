from ml_model import model
from neuralop.datasets import load_darcy_flow_small
import matplotlib.pyplot as plt
import numpy as np
import torch
np.random.seed(42)
seed = 42  
torch.manual_seed(seed)
# 加载数据
train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000, batch_size=32,
        test_resolutions=[16, 32], n_tests=[100, 50],
        test_batch_sizes=[32, 32],
        positional_encoding=True
)

# 定义似然函数
def likelihood(observed_data, simulated_data):
    # print(observed_data.shape,simulated_data.shape)
    return -torch.sum((observed_data-simulated_data)**2)

# 定义MCMC采样函数
def mcmc_sampling(initial_state, observed_data, num_samples, noise_level=0.1):
    current_state = initial_state
    state_sum = current_state.clone().detach().float().zero_()
    best_state = current_state.clone().detach()
    best_likelihood=likelihood(observed_data,model(torch.tensor(initial_state, dtype=torch.float32).unsqueeze(0))).item()
    with torch.no_grad():
        for _ in range(num_samples):
            proposed_state = current_state.clone().detach()
            random_numbers = torch.rand(proposed_state[0].shape)
            probability=0.01
            proposed_state[0] = torch.where(random_numbers < probability, 1 - proposed_state[0], proposed_state[0])
            # proposed_state[0]=proposed_state[0]+np.random.normal(0, noise_level, proposed_state[0].shape)
            # proposed_state[0] = torch.clamp(proposed_state[0], 0.0, 1.0)
            proposed_state_tensor = torch.tensor(proposed_state, dtype=torch.float32).unsqueeze(0)
            # 使用模型预测模拟数据
            simulated_data = model(proposed_state_tensor)

            current_state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)
            current_simulated_data = model(current_state_tensor)
            # print(torch.sum((current_state-proposed_state)**2))
            # # 计算当前和提议状态的似然
            current_likelihood = likelihood(observed_data, current_simulated_data)
            proposed_likelihood = likelihood(observed_data, simulated_data)
            print(current_likelihood)
            print(proposed_likelihood)
            if proposed_likelihood > best_likelihood:
                best_likelihood = proposed_likelihood.item()
                best_state = proposed_state.clone().detach()
            print(best_likelihood)
            # # 计算接受率
            acceptance_ratio = np.exp((proposed_likelihood.detach().numpy() - current_likelihood.detach().numpy()))# / min(1.0, np.abs(current_likelihood.detach().numpy())))
            print(acceptance_ratio)
            if np.random.rand() < acceptance_ratio:
                current_state = proposed_state

            # samples.append(np.array(current_state))
            if _>=num_samples//2:
                state_sum += current_state
    
    return state_sum/num_samples*2

# 获取测试样本
test_samples = test_loaders[32].dataset
index=0
test_data = test_samples[index]
data = data_processor.preprocess(test_data, batched=False)
# 初始状态和观测数据
x0 = data['x']
observed_data = data['y']
print(likelihood(observed_data,model(torch.tensor(x0, dtype=torch.float32).unsqueeze(0))).item())

# 设置初始条件和采样数量
initial_state=x0
channels, height, width = initial_state.shape

initial_state[0]=(torch.rand(height, width)>= 0.5).float()
# initial_state[0]=torch.rand(height, width)
num_samples = 10000  # 样本数量可以适当增大

# 运行MCMC采样
ave_sample = mcmc_sampling(initial_state, observed_data, num_samples)

# 绘图展示结果
fig = plt.figure(figsize=(7, 9))
mcmc_pred = ave_sample
data = test_samples[index]
data = data_processor.preprocess(data, batched=False)
x = data['x']
y = data['y']

# 使用MCMC结果中的最后一个样本进行预测
ax = fig.add_subplot(4, 1,  1)
ax.imshow(x[0], cmap='gray')
ax.set_title('Input x')
plt.xticks([], [])
plt.yticks([], [])

ax = fig.add_subplot(4, 1,  2)
ax.imshow(y.squeeze())
ax.set_title('Ground-truth y')
plt.xticks([], [])
plt.yticks([], [])

ax = fig.add_subplot(4, 1, 3)
ax.imshow(mcmc_pred[0].squeeze(), cmap='gray')
ax.set_title('MCMC prediction')
plt.xticks([], [])
plt.yticks([], [])

pred_in=torch.tensor(mcmc_pred, dtype=torch.float32).unsqueeze(0)
out = model(pred_in)
ax = fig.add_subplot(4, 1, 4)
ax.imshow(out.squeeze().detach().numpy())
ax.set_title('Model prediction')
plt.xticks([], [])
plt.yticks([], [])

# print(x[0],mcmc_pred[0])
print(torch.sum((x[0]-mcmc_pred[0])**2))

fig.suptitle('Inputs, Ground-truth and MCMC Predictions', y=0.98)
plt.tight_layout()
plt.show()