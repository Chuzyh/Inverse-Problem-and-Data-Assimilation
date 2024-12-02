import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numerical_model import lorenz96,lorenz96_twoscale
from ml_model import nn
# Simulation parameters
dt = 0.05
n_steps = 100
t_eval = np.arange(0, n_steps * dt, dt)
x0 = np.random.randn(40)  # Initial state for the Lorenz 96 model

# Two-scale initial condition (main + subgrid states)
n_subgrid = 3
x0_twoscale = np.hstack((x0, np.random.randn(n_subgrid * 40)))

# Integrate the Lorenz 96 model
sol_single = solve_ivp(
    lorenz96, [0, n_steps * dt], x0, t_eval=t_eval, args=(8,)
).y.T

# Integrate the Lorenz 96 two-scale model
sol_twoscale = solve_ivp(
    lorenz96_twoscale,
    [0, n_steps * dt],
    x0_twoscale,
    t_eval=t_eval,method="RK45",
    args=(40, n_subgrid, 8, None)
).y.T

# Extract only the main scale states from the two-scale model
sol_twoscale_main = sol_twoscale[:, :40]

# Debugging
print("sol_single shape:", sol_single.shape)         # (1000, 40)
print("sol_twoscale shape:", sol_twoscale.shape)     # (1000, 40 + n_subgrid * 40)
print("sol_twoscale_main shape:", sol_twoscale_main.shape)  # (1000, 40)
x_ml = np.zeros((n_steps, 40))
x_ml[0] = x0
x = x0
for i in range(1, n_steps):
    x = nn._smodel.predict(x.reshape((1, 40, 1)))[0, :, 0]
    x_ml[i] = x

dimension = 0  # 第一个维度
time = np.arange(n_steps) * dt  # 时间轴

plt.figure(figsize=(10, 6))
plt.plot(time, sol_single[:, dimension], label='single', color='blue')
plt.plot(time, sol_twoscale_main[:, dimension], label='two scale', color='orange')
plt.plot(time, x_ml[:, dimension], label='ML Model', color='green')
plt.xlabel('Time')
plt.ylabel(f'Dimension {dimension+1} Value')
plt.title(f'Variation of Dimension {dimension+1}')
plt.legend()
plt.grid()
plt.savefig(f"dimension_{dimension+1}_variation.pdf")
plt.show()
# Compute RMSE at each time step
# rmse = np.sqrt(((sol_single - sol_twoscale_main) ** 2).mean(axis=1))
# rmse_ml = np.sqrt(((x_ml - sol_twoscale_main) ** 2).mean(axis=1))
# rmse_sin = np.sqrt(((x_ml - sol_single) ** 2).mean(axis=1))

# # Plot the RMSE
# plt.figure(figsize=(10, 6))
# plt.plot(t_eval, rmse, label='RMSE (Lorenz 96 vs Two-scale)')
# plt.plot(t_eval, rmse_ml, label='RMSE (ML vs Two-scale)')
# plt.plot(t_eval, rmse_sin, label='RMSE (ML vs Lorenz 96)')
# plt.xlabel('Time')
# plt.ylabel('RMSE')
# plt.title('Error between Lorenz 96 and Two-scale Model')
# plt.legend()
# plt.grid()
# plt.savefig("lorenz96_error.pdf")
# plt.show()
