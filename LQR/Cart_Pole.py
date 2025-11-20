# Import packages:
import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt

def cart_pole(Q, R, dt, n_tsteps):
    # Define linear system:
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, -0.1818, 0],
        [0, 0, 0, 1],
        [0, 0, 2.6727, 0]
    ])

    B = np.array([[0], [1.8182], [0], [-2.7273]])

    # Solve ARE:
    P = solve_continuous_are(A, B, Q, R)

    # Define gain matrix K:
    K = np.linalg.inv(R) @ B.T @ P

    # Initialize:
    x = np.zeros((4, n_tsteps))
    u = np.zeros(n_tsteps)
    x[:, 0] = np.array([0.1, 0.0, 0.2, 0.0]) # Initial perturbation:

    # Simulate forwards (Euler integration):
    for index in range(n_tsteps - 1):
        u[index] = -K @ x[:, index]
        xdot = A @ x[:, index] + B[:, 0] * u[index]
        x[:, index + 1] = x[:, index] + dt * xdot

    # Return simulated states:
    return x, u


# Problem parameters:
dt = 0.01
T_tot = 10
n_tsteps = int(T_tot / dt)

# Run several simulations with different weight matrices:
x1, u1 = cart_pole(1 * np.diag([1, 1, 10, 1]), 1 * np.array([[0.1]]), dt, n_tsteps)
x2, u2= cart_pole(0.5 * np.diag([1, 1, 10, 1]), 5 * np.array([[0.1]]), dt, n_tsteps)
x3, u3 = cart_pole(0.1 * np.diag([1, 1, 10, 1]), 10 * np.array([[1]]), dt, n_tsteps)

# === Plot results ===
t = np.linspace(0, T_tot, n_tsteps)
fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
plt.suptitle("Cart-Pole LQR Response")

# Plot position:
axs[0].plot(t, x1[0, :], label='Performance')
axs[0].plot(t, x2[0, :], label='Balanced')
axs[0].plot(t, x3[0, :], label='Control Effort')
axs[0].axhline(0, color='k', linestyle='--')
axs[0].set_ylabel("Position (m)")

# Plot angle:
axs[1].plot(t, np.rad2deg(x1[2, :]))
axs[1].plot(t, np.rad2deg(x2[2, :]))
axs[1].plot(t, np.rad2deg(x3[2, :]))
axs[1].axhline(0, color='k', linestyle='--')
axs[1].set_ylabel("\u0398 (deg)")
axs[1].legend(title="Priority")

# Plot control:
axs[2].plot(t, u1, label='Performance')
axs[2].plot(t, u2, label='Balanced')
axs[2].plot(t, u3, label='Control Effort')
axs[2].set_ylabel("Control (N)")
axs[2].set_xlabel("Time (s)")
axs[2].axhline(0, color='k', linestyle='--')

# Plot:
plt.xlim([0, T_tot])
plt.show()
