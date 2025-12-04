"""
Driver function that defines the system

--- Takes the following nonlinear problem ---
Consider a 2D simple aerodynamic problem modelled by:
x = [px, py, vx, vy] --> State Vector
u = [ux, uy] --> Control Vector
\dot{px} = vx
\dot{py} = vy
\dot{vx} = ux - c * vx * sqrt(vx^2 + vy^2)
\dot{vy} = uy - c * vy * sqrt(vx^2 + vy^2)
where c is a drag constant (e.g. 0.4)

--- Describing MPC Problem ---
As this problem is nonlinear, we can linearize the problem at any given point as:
A_k = df/dx|_{x=xk} and B_k = df/du|_{u=uk}

This can then be evaluated as a regular linear system in which:
\dot{x} = Ax + Bu
y = Cx

which will again need to be discretizes such that
x_{k+1} = Ax_{k} + Bu_{k}
y_{k} = Cx_{k}

Since we already have this discretization implementation and the following MPC architecture, the
only addition needed is a function to linearize at each step.

"""

# Import packages:
import numpy as np
import matplotlib.pyplot as plt
from src.MPC import MPC
from src.utils import *

# Define MPC parameters:
f = 50  # Prediction horizon
v = 50  # Control horizon

# Simulation parameters:
dt = 0.01  # s
t_end = 20  # s
n_tsteps = int(t_end/dt)
time = np.linspace(0, t_end, n_tsteps)
time_ctrl = np.linspace(0, t_end, n_tsteps - f)

#################### CHANGE THIS FOR NEW SYSTEM ####################

# Import interpolated system matrices:
interp_a, interp_b = train_rbf()
A_cts = interp_a([[15, 6, 6]]).reshape(12, 12)
B_cts = interp_b([[15, 6, 6]]).reshape(12, 4)

# Define observability matrix (fully observable)
C_cts = np.eye(12)

# Initial conditions:
x0 = np.array([15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Define weight scales:
Q0_scaler = 1e-4
Q_scaler = 1e-2
P_scaler = 1e2

# Define control bounds:
u_max = np.deg2rad(np.array([20, 20, 20, 20])).T
u_min = -u_max

####################################################################

# --- Define weight matrices:
# Penalize current size of u:
Q0 = np.array([[0.00001, 0, 0, 0],
               [0, 0.00001, 0, 0],
               [0, 0, 0.0001, 0],
               [0, 0, 0, 0.00001]])
Q0 *= Q0_scaler

# Penalize difference between u values at subsequent steps:
Q = np.array([[0.0001, 0, 0, 0],
              [0, 0.0001, 0, 0],
              [0, 0, 0.0001, 0],
              [0, 0, 0, 0.0001]])
Q *= Q_scaler

# Penalize error in objective:
P = np.array([[10., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 10., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 10., 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 10., 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 10., 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 10., 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 10., 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 10., 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 10., 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 10., 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10., 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10.]])
P *= P_scaler

# --- Develop MPC problem ---
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
    if i == 0:
        Q = Q0
    else:
        Q = Q
    W2[i * m:(i + 1) * m, i * m:(i + 1) * m] = Q

# W3 matrix:
W3 = W1.T @ W2 @ W1

# W4 matrix
W4 = np.zeros(shape=(f * r, f * r))
for i in range(f):
    W4[i * r:(i + 1) * r, i * r:(i + 1) * r] = P

# --- Desired trajectories ---
traj = np.zeros((n_tsteps, 12))
traj[:, 0] = np.linspace(15, 10, n_tsteps)
traj[:, 8] = np.linspace(0, 15 * t_end, n_tsteps)

# --- Use MPC controller ---
# Build MPC object:
mpc = MPC(A, B, C, f, v, W3, W4, x0, traj, u_min, u_max, 'linear') # Already technically linearized

# Use controller:
for i in range(n_tsteps - f):
    mpc.control_inputs()

# --- Unpack, plot ---
# Extract state estimates:
y_des = []
y = []
ctrl = []
for j in np.arange(n_tsteps - f):
    y.append(mpc.outputs[j][ :, 0])
    y_des.append(traj[j, :])
    ctrl.append(mpc.inputs[j][:, 0])

# Switch to arrays:
y = np.array(y)
y_des = np.array(y_des)
ctrl = np.array(ctrl)

# Plot graphs:
plot_graphs(time_ctrl, y, y_des, ctrl, 'AVL', np.array([u_min, u_max]))

# Show results:
plt.show()
