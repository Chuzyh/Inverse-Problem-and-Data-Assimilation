import numpy as np

def lorenz96(t, x, F=8):
    """Lorenz 96 model with constant forcing"""
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return dxdt

def lorenz96_twoscale(t, u, N=40, n=5, F=8, p=None):
    if p is None:
        p = {"h": 1.0, "c": 10.0, "b": 10.0}  # Sample parameter defaults, set as needed

    dx = np.zeros(N)
    dy = np.zeros((n, N))

    u = u.reshape(n + 1, N)
    x = u[0, :]
    y = u[1:, :]

    for i in range(N):
        dx[i] = ((x[(i + 1) % N] - x[(i - 2) % N]) * x[(i - 1) % N] 
                 - x[i] + F - p["h"] * p["c"] / p["b"] * np.sum(y[:, i]))

        for j in range(n):
            if j == n - 1:
                jp1, jp2, jm1 = 0, 1, n - 2
                ip1, ip2, im1 = (i + 1) % N, (i + 1) % N, i
            elif j == n - 2:
                jp1, jp2, jm1 = n - 1, 0, n - 3
                ip1, ip2, im1 = i, (i + 1) % N, i
            elif j == 0:
                jp1, jp2, jm1 = 1, 2, n - 1
                ip1, ip2, im1 = i, i, (i - 1) % N
            else:
                jp1, jp2, jm1 = j + 1, j + 2, j - 1
                ip1, ip2, im1 = i, i, i

            dy[j, i] = (p["c"] * p["b"] * y[jp1, ip1] * (y[jm1, im1] - y[jp2, ip2]) 
                        - p["c"] * y[j, i] + p["h"] * p["c"] / p["b"] * x[i])

    du = np.hstack((dx, dy.flatten()))

    return du
