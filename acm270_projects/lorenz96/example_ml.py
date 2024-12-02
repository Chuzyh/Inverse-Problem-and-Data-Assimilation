from ml_model import nn
from numerical_model import lorenz96
import pymc as pm
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import arviz as az

# Time step for the ML model; use the same for the numerical integration
dt = 0.05
n_steps = 4
dim=40

x0_true = np.random.randn(dim)  # 实际初始状态
x_phys = solve_ivp(lorenz96, [0, n_steps * dt], x0_true, t_eval=np.arange(0.0, n_steps * dt, dt)).y.T
print(x_phys)
def likelihood(x0_np):
    x = x0_np
    x_ml = np.zeros((n_steps, dim))
    x_ml[0] = x0_np
    for i in range(1, n_steps):
        x = nn._smodel.predict(x.reshape((1, dim, 1)))[0, :, 0]
        x_ml[i] = x
    return x_ml
X_ML=likelihood(x0_true)
print(X_ML)
plt.plot(np.sqrt(((X_ML - x_phys)**2).mean(axis=1)))

plt.savefig("rmse.pdf")
# 使用 MCMC 方法估计初始状态 x0
with pm.Model() as model:
    # 定义初始条件
    x0 = pm.Normal('x0', mu=0, sigma=1, shape=(dim,))
    
    # 定义观测数据的预测函数
    def likelihood_fn(x0):
        x0_np = x0.eval() if hasattr(x0, 'eval') else x0
        x = x0_np
        x_ml = np.zeros((n_steps, dim))
        x_ml[0] = x0_np
        for i in range(1, n_steps):
            x = nn._smodel.predict(x.reshape((1, dim, 1)))[0, :, 0]
            x_ml[i] = x
        return x_ml

    # 计算预测均值

    # 观测模型，使用观测数据
    
    obs = pm.Normal('obs', mu=likelihood_fn(x0), sigma=0.1, observed=x_phys)

    # 使用 Metropolis-Hastings 进行采样
    step = pm.Metropolis()

    # 进行采样
    trace = pm.sample(100000, step=step, return_inferencedata=True)

# 绘制后验分布
pm.plot_trace(trace)
plt.savefig("trace_plot.pdf")
print(az.summary(trace))
print(x0_true)
# 计算后验均值
# x0_estimated = trace['x0'].mean(axis=0)
# print("Estimated initial state x0:", x0_estimated)