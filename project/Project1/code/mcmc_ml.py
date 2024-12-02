import numpy as np
from ml_model import nn
from numerical_model import lorenz96
dim=40
# Lorenz96 方程的定义
def lorenz96(x, t, F):
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return dxdt

# 模型的运行函数，F固定为8
observe_every=1
observed_indices = np.arange(0, dim, observe_every)  # 定义观测维度的索引
    
def run_lorenz96_model(initial_conditions, F=8, time=0.2, num_steps=4):
    x0_np = initial_conditions.eval() if hasattr(initial_conditions, 'eval') else initial_conditions
    x = x0_np
    # x_ml = np.zeros((num_steps, dim))
    # x_ml[0] = x0_np
    for i in range(1, num_steps):
        x = nn._smodel.predict(x.reshape((1, dim, 1)))[0, :, 0]
        # x_ml[i] = x
    # return x_ml[0:]
    return x[observed_indices]
# 似然函数，基于观测数据和模拟数据之间的差异
def likelihood(observed_data, simulated_data):
    return -0.5 * np.sum((observed_data - simulated_data) ** 2)

# MCMC 采样函数
def mcmc_sampling(initial_state, observed_data, num_samples, noise_level=0.01):
    current_state = initial_state
    samples = []
    
    for _ in range(num_samples):
        # 为当前状态生成一个小的随机扰动
        proposed_state = current_state + np.random.normal(0, noise_level, current_state.shape)
        
        # 计算模拟数据
        simulated_data = run_lorenz96_model(proposed_state) 
        # 计算似然
        current_likelihood = likelihood(observed_data, run_lorenz96_model(current_state))
        proposed_likelihood = likelihood(observed_data, simulated_data)
        print(current_likelihood,proposed_likelihood)
        # 计算接受率
        acceptance_ratio = np.exp((proposed_likelihood - current_likelihood)/min(1.0,np.abs(current_likelihood)))
        
        print(acceptance_ratio)
        # 按照接受率决定是否接受新状态
        if np.random.rand() < acceptance_ratio:
            current_state = proposed_state
        
        samples.append(current_state)
        
    return np.array(samples)

# 示例数据：假设观测到了100个时间点的状态
np.random.seed(42)  # 设置随机种子，确保结果可复现
x0 = np.random.uniform(-1, 1, dim)
phy_data =  run_lorenz96_model(x0)
observed_data = phy_data#+np.random.normal(0, 0.01, phy_data.shape) # 假设有 40 维观测数据

# 初始条件（40维变量，F 固定为 8）
initial_state = np.random.uniform(-1, 1, dim)
# print(likelihood(observed_data,run_lorenz96_model(initial_state)))
print(x0,initial_state)
# MCMC 采样的样本数
num_samples = 3000000

# 运行 MCMC 采样
samples = mcmc_sampling(initial_state, observed_data, num_samples)

# 打印
print('sample mean ',samples[num_samples//2:].mean(0))
print('ground truth ',x0)
print('init ',initial_state)
print(np.sum((samples[num_samples//2:].mean(0)-x0)**2))
print(np.sum((initial_state-x0)**2))

import matplotlib.pyplot as plt
import seaborn as sns

# 可视化 MCMC 采样的分布
def plot_sample_distributions(samples, ground_truth, dim=40):
    plt.figure(figsize=(12, 8))
    
    for i in range(min(dim, 5)):  # 只展示前5个维度
        plt.subplot(2, 3, i + 1)
        sns.kdeplot(samples[:, i], label='MCMC Samples', color='blue', fill=True)
        plt.axvline(ground_truth[i], color='red', linestyle='--', label='Ground Truth')
        plt.title(f'Dimension {i+1}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

# 散点图：真实初始状态 vs 采样均值
def plot_scatter_samples(samples_mean, ground_truth):
    plt.figure(figsize=(6, 6))
    plt.scatter(range(len(ground_truth)), ground_truth, label='Ground Truth', color='red')
    plt.scatter(range(len(samples_mean)), samples_mean, label='Sample Mean', color='blue', alpha=0.6)
    plt.title('Sample Mean vs Ground Truth')
    plt.xlabel('Dimension')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# 调用可视化函数
plot_sample_distributions(samples[num_samples//2:], x0, dim=40)
plot_scatter_samples(samples[num_samples//2:].mean(0), x0)
