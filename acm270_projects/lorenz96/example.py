from ml_model import nn
from numerical_model import lorenz96

from scipy.integrate import odeint
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Time step for the ML model; use the same for the numerical integration
dt = 0.05
n_steps = 100

# Generate a random state
x0 = np.random.randn(40)

x = x0
x_ml = np.zeros((n_steps, 40))
x_ml[0] = x0
for i in range(1, n_steps):
    x = nn._smodel.predict(x.reshape((1, 40, 1)))[0, :, 0]
    x_ml[i] = x
def lorenz96_(x, t, F):
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return dxdt

# 模型的运行函数，F固定为8
def run_lorenz96_model(initial_conditions, F=8, time=dt*n_steps, num_steps=n_steps):
    t = np.linspace(0, time, num_steps)  # 时间跨度
    return odeint(lorenz96_, initial_conditions, t, args=(F,))

x_phys = solve_ivp(lorenz96, [0, n_steps*dt], x0, t_eval=np.arange(0.0, n_steps*dt, dt)).y.T
# x_phys = run_lorenz96_model(x0)

# Plot the RMSE between the physical and ML forecast
plt.plot(np.sqrt(((x_ml - x_phys)**2).mean(axis=1)))
x_phys = run_lorenz96_model(x0)
plt.plot(np.sqrt(((x_ml - x_phys)**2).mean(axis=1)))

plt.savefig("rmse.pdf")
