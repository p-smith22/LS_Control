# Import packages:
import numpy as np
from rbf.interpolate import RBFInterpolant
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Train RBF for A and B matrices:
def train_rbf():

    # Load data:
    a = np.load('./AVL_Data/A_trainer_5_large.npy')
    b = np.load('./AVL_Data/B_trainer_5_large.npy')
    trainer_vals = np.load('./AVL_Data/sobolsequence_5_large.npy')

    # Formatting:
    iterations = len(trainer_vals[:, 0])
    a_reshape = a.reshape(iterations, -1)
    b_reshape = b.reshape(iterations, -1)

    # Set interpolated matrices:
    interp_a = RBFInterpolant(trainer_vals, a_reshape)
    interp_b = RBFInterpolant(trainer_vals, b_reshape)
    return interp_a, interp_b

# Translate continuous matrices to continuous (e.g. A --> e**(A * dt)) for discrete linear system:
def discretize_system(a_cts, b_cts, dt):

    n = a_cts.shape[0]
    m = b_cts.shape[1]

    # Build augmented matrix [A, B; 0, 0]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = a_cts * dt
    M[:n, n:] = b_cts * dt

    # Exact discretization using matrix exponential
    exp_m = expm(M)

    a_disc = exp_m[:n, :n]
    b_disc = exp_m[:n, n:]

    return a_disc, b_disc

# Calculate controllability matrix:
def calc_c(a, b, steps):
    n = a.shape[0]
    m = b.shape[1]
    c = np.zeros((n, steps * m))
    for i in range(steps):
        c[:, i * m:(i + 1) * m] = np.linalg.matrix_power(a, steps - 1 - i) @ b
    return c


# Problem parameters:
x_0 = np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
x_f = np.array([15, 0, 0, 0, 0, 0, 0, 0, 100, 50, 50, 0])

# Time parameters
dt = 0.1
T_tot = 10.0
n_tsteps = int(T_tot / dt)

# Load interpolators:
A_interp, B_interp = train_rbf()

# Fetch system matrices (nominal configuration):
A_cts = A_interp([[15, 6, 0]]).reshape(-1, 12, 12)[0]
B_cts = B_interp([[15, 6, 0]]).reshape(-1, 12, 4)[0]

# Discretize the system:
A, B = discretize_system(A_cts, B_cts, dt)

# Build controllability matrix
C = calc_c(A, B, n_tsteps)

# Check controllability:
rank = np.linalg.matrix_rank(C)
if rank != A.shape[0]:
    raise Exception("ERROR: Singular Matrix")

# Solve for control inputs using pseudoinverse:
RHS = x_f - np.linalg.matrix_power(A, n_tsteps) @ x_0
U_flat = np.linalg.pinv(C) @ RHS
U = U_flat.reshape(n_tsteps, B.shape[1])

# Simulate forwards in time (can simply step because we discretized the matrices):
x = np.zeros((A.shape[0], n_tsteps + 1))
x[:, 0] = x_0
for k in range(n_tsteps):
    x[:, k + 1] = A @ x[:, k] + (B @ U[k]).flatten()

# Plot trajectory:
time = np.arange(n_tsteps + 1) * dt

# Plot position:
fig, ax1 = plt.subplots(3, 1, figsize=(10, 8))
fig.suptitle('Position States', fontsize=14, fontweight='bold')
position_labels = ['x (m)', 'y (m)', 'z (m)']
position_idx = [8, 9, 10]
for i, idx in enumerate(position_idx):
    ax1[i].plot(time, x[idx, :], linewidth=2, color='blue')
    ax1[i].axhline(x_0[idx], color='green', linestyle='--', label='Initial', linewidth=1.5)
    ax1[i].axhline(x_f[idx], color='red', linestyle='--', label='Target', linewidth=1.5)
    ax1[i].set_ylabel(position_labels[i], fontsize=12)
    ax1[i].grid(True, alpha=0.3)
    ax1[i].legend()
ax1[-1].set_xlabel('Time (s)', fontsize=12)
plt.tight_layout()

# Plot velocities:
fig, ax2 = plt.subplots(3, 1, figsize=(10, 8))
fig.suptitle('Velocity States', fontsize=14, fontweight='bold')
velocity_labels = ['u (m/s)', 'v (m/s)', 'w (m/s)']
velocity_idx = [0, 4, 1]
for i, idx in enumerate(velocity_idx):
    ax2[i].plot(time, x[idx, :], linewidth=2, color='blue')
    ax2[i].axhline(x_0[idx], color='green', linestyle='--', label='Initial', linewidth=1.5)
    ax2[i].axhline(x_f[idx], color='red', linestyle='--', label='Target', linewidth=1.5)
    ax2[i].set_ylabel(velocity_labels[i], fontsize=12)
    ax2[i].grid(True, alpha=0.3)
    ax2[i].legend()
ax2[-1].set_xlabel('Time (s)', fontsize=12)
plt.tight_layout()

# Plot Euler angles:
fig, ax3 = plt.subplots(3, 1, figsize=(10, 8))
fig.suptitle('Euler Angles', fontsize=14, fontweight='bold')
angle_labels = ['\u0398 (deg)', '\u03A6 (deg)', '\u03A8 (deg)']
angle_idx = [3, 7, 11]
for i, idx in enumerate(angle_idx):
    ax3[i].plot(time, np.rad2deg(x[idx, :]), linewidth=2, color='blue')
    ax3[i].axhline(np.rad2deg(x_0[idx]), color='green', linestyle='--', label='Initial', linewidth=1.5)
    ax3[i].axhline(np.rad2deg(x_f[idx]), color='red', linestyle='--', label='Target', linewidth=1.5)
    ax3[i].set_ylabel(angle_labels[i], fontsize=12)
    ax3[i].grid(True, alpha=0.3)
    ax3[i].legend()
ax3[-1].set_xlabel('Time (s)', fontsize=12)
plt.tight_layout()

# Plot Euler rates:
fig, ax4 = plt.subplots(3, 1, figsize=(10, 8))
fig.suptitle('Angular Rates', fontsize=14, fontweight='bold')
rate_labels = ['q (deg/s)', 'p (deg/s)', 'r (deg rate)']
rate_idx = [2, 5, 6]
for i, idx in enumerate(rate_idx):
    ax4[i].plot(time, np.rad2deg(x[idx, :]), linewidth=2, color='blue')
    ax4[i].axhline(np.rad2deg(x_0[idx]), color='green', linestyle='--', label='Initial', linewidth=1.5)
    ax4[i].axhline(np.rad2deg(x_f[idx]), color='red', linestyle='--', label='Target', linewidth=1.5)
    ax4[i].set_ylabel(rate_labels[i], fontsize=12)
    ax4[i].grid(True, alpha=0.3)
    ax4[i].legend()
ax4[-1].set_xlabel('Time (s)', fontsize=12)
plt.tight_layout()

# Plot controls:
fig, ax5 = plt.subplots(4, 1, figsize=(10, 10))
fig.suptitle('Control Inputs', fontsize=14, fontweight='bold')
control_labels = ['Camber (deg)', 'Aileron (deg)', 'Elevator (deg)', 'Rudder (deg)']
for i in range(U.shape[1]):
    ax5[i].plot(time[:-1], np.rad2deg(U[:, i]), linewidth=2, color='red')
    ax5[i].set_ylabel(control_labels[i], fontsize=12)
    ax5[i].grid(True, alpha=0.3)
ax5[-1].set_xlabel('Time (s)', fontsize=12)
plt.tight_layout()

# Show graphs:
plt.show()
