import numpy as np
from scipy.integrate import solve_ivp
# Lorenz96 方程的定义
def lorenz96(t, x, F):
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return dxdt
dim=40
observe_every=2
observed_indices = np.arange(0, dim, observe_every) 
# 模型的运行函数，F固定为8
def run_lorenz96_model(initial_conditions, F=8, dt=0.01, n_steps=20, observed_indices=None):
    
    # 时间跨度设置
    t_eval = np.arange(0.0, (n_steps+1) * dt, dt)

    # 使用 solve_ivp 求解
    solution = solve_ivp(
        lorenz96,                # 方程
        [0, n_steps * dt],       # 时间区间
        initial_conditions,      # 初始条件
        t_eval=t_eval,           # 时间采样点
        args=(F,),               # 额外参数传递给 lorenz96
        method='RK45'            # 数值方法
    )
    
    # 提取最后一个时间点的指定变量
    result = solution.y.T[-1]
    if observed_indices is not None:
        result = result[observed_indices]
    
    return result

# 似然函数，基于观测数据和模拟数据之间的差异
def likelihood(observed_data, simulated_data):
    return -0.5 * np.sum((observed_data - simulated_data) ** 2)

# MCMC 采样函数
def mcmc_sampling(initial_state, observed_data, num_samples, noise_level=0.1):
    current_state = initial_state
    samples = []
    
    for _ in range(num_samples):
        # 为当前状态生成一个小的随机扰动
        proposed_state = current_state + np.random.normal(0, noise_level, current_state.shape)
        
        # 计算模拟数据
        simulated_data = run_lorenz96_model(proposed_state) 
        # 计算似然
        current_likelihood = likelihood(observed_data[observed_indices], run_lorenz96_model(current_state,observed_indices=observed_indices))
        proposed_likelihood = likelihood(observed_data[observed_indices], simulated_data[observed_indices])
        print(current_likelihood,proposed_likelihood)
        # 计算接受率
        acceptance_ratio = np.exp((proposed_likelihood - current_likelihood)/min(1.0,np.abs(current_likelihood)))
        
        print(acceptance_ratio)
        # 按照接受率决定是否接受新状态
        if np.random.rand() < acceptance_ratio:
            current_state = proposed_state
        
        samples.append(current_state)
        
    return np.array(samples)

np.random.seed(42)  # 设置随机种子，确保结果可复现
x0 = np.zeros(dim)#np.random.uniform(-1, 1, dim)
phy_data =  run_lorenz96_model(x0)
observed_data = phy_data # 假设有 dim 维观测数据

# 初始条件（40维变量，F 固定为 8）
initial_state = np.random.uniform(-1, 1, dim)
print(likelihood(observed_data,run_lorenz96_model(initial_state)))
print(x0,initial_state)
# MCMC 采样的样本数
num_samples_burn = 20000
num_samples = 2000

# 运行 MCMC 采样
samples = mcmc_sampling(initial_state, observed_data, num_samples_burn+num_samples)
print(samples.shape)
# 打印
print('sample mean ',samples[num_samples_burn+1:].mean(0))
print('ground truth ',x0)
print('init ',initial_state)
print(np.sum((samples[num_samples_burn+1:].mean(0)-x0)**2))
print(np.sum((initial_state-x0)**2))

import matplotlib.pyplot as plt
import seaborn as sns

# 可视化 MCMC 采样的分布
def plot_sample_distributions(samples, ground_truth, dim):
    plt.figure(figsize=(12, 8))
    
    for i in range(dim):
        sns.kdeplot(samples[num_samples_burn+1:, i], label=f'Dimension {i+1}', fill=True, alpha=0.4)
        plt.axvline(ground_truth[i], linestyle='--', label=f'Ground Truth {i+1}')
    
    plt.title('Sample Distributions Across Dimensions')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


# 调用可视化函数
plot_sample_distributions(samples[num_samples//2:], x0, dim=dim)