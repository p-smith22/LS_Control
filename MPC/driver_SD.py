"""
Driver function that defines the system

Takes a continuous system
x_dot = Ax + Bu
y = Cx
and discretizes such that
x_{k+1} = Ax_{k} + Bu_{k}
y_{k} = Cx_{k}

Therefore, for nw systems, simply change the definition of the problem
to your cts system and the rest will adjust automatically
"""

# Import packages:
import numpy as np
import matplotlib.pyplot as plt
from src.MPC import MPC
from src.utils import *

# Define MPC parameters:
f = 20  # Prediction horizon
v = 20  # Control horizon

# Simulation parameters:
dt = 0.01 # s
t_end = 3  # s
n_tsteps = int(t_end/dt)
time = np.linspace(0, t_end, n_tsteps)
time_ctrl = np.linspace(0, t_end, n_tsteps - f)

#################### CHANGE THIS FOR NEW SYSTEM ####################

# Masses:
m1 = 2
m2 = 2

# Springs:
k1 = 100
k2 = 200

# Dashpot:
d1 = 1
d2 = 5

# Define system matrices (continuous):
A_cts = np.matrix([[    0,                1,            0,        0   ],
                   [-(k1 + k2) / m1, -(d1 + d2) / m1, k2 / m1, d2 / m1],
                   [    0,                0,            0,        1   ],
                   [  k2 / m2,         d2 / m2,      -k2 / m2, -d2 / m2]])
B_cts = np.matrix([[0], [0], [0], [1 / m2]])
C_cts = np.matrix([[1, 0, 0, 0]])

# Initial conditions:
x0 = np.zeros((4, 1))

# Control weight matrices:
m = B_cts.shape[1]
Q = np.zeros((m, m , v))
Q[:, :, 0] = 1e-5  # Penalizes size of current u
Q[:, :, 1:] = 1e-2  # Penalizes difference between subsequent u values

# State weight matrices:
r = C_cts.shape[0]
P = np.zeros((r, r, f))
P[:, :, :] = 1e2

# Set control limits (can set these as None if you only want one or no bounds):
u_max = np.array([5])
u_min = -u_max

# Optional: NO BOUNDS:
# u_max = None
# u_min = None

####################################################################

# Fetch dimensions:
r = C_cts.shape[0]
m = B_cts.shape[1]
n = A_cts.shape[1]

# Discretize matrices:
A, B = discretize_system(A_cts, B_cts, dt)
C = C_cts

# W1 matrix:
W1 = np.zeros(shape=(v * m, v * m))
for i in range(v):
    W1[i * m:(i + 1) * m, i * m:(i + 1) * m] = np.eye(m, m)
    if i > 0:
        W1[i * m:(i + 1) * m, (i - 1) * m:i * m] = -np.eye(m, m)

# W2 matrix:
W2 = np.zeros(shape=(v * m, v * m))
for i in range(v):
    W2[i * m:(i + 1) * m, i * m:(i + 1) * m] = Q[:, :, i]

# W3 matrix:
W3 = W1.T @ W2 @ W1

# W4 matrix
W4 = np.zeros(shape=(f * r, f * r))
for i in range(f):
    W4[i * r:(i + 1) * r, i * r:(i + 1) * r] = P[:, :, i]

# --- Test trajectories ---
# Exponential trajectory:
# traj = np.ones(time) - np.exp(-0.01 * time)
# traj = np.reshape(traj, (n_tsteps, 1))

# Pulse trajectory:
# traj = np.zeros((n_tsteps, 1))
# third = int(n_tsteps/3)
# traj[0:third, :] = np.ones((third, 1))
# traj[-third:, :] = np.ones((third, 1))

# Step trajectory:
traj = np.zeros((n_tsteps, 1))
traj[:, 0]= (2.5 / 100) * np.ones((n_tsteps,1)).flatten()

# Build MPC object:
mpc = MPC(A, B, C, f, v, W3, W4, x0, traj, u_min, u_max)

# Use controller:
for i in range(n_tsteps - f):
    mpc.control_inputs()

# Extract state estimates:
y_des = []
y = []
ctrl = []
for j in np.arange(n_tsteps - f):
    y.append(mpc.outputs[j][ :, 0])
    y_des.append(traj[j, :])
    ctrl.append(mpc.inputs[j][0, 0])

# Switch to arrays:
y = np.array(y)
y_des = np.array(y_des)
ctrl = np.array(ctrl)

# Plot graphs:
plot_graphs(time_ctrl, y, y_des, ctrl, 'spring_dashpot', np.array([u_min, u_max]))

# Show results:
plt.show()
