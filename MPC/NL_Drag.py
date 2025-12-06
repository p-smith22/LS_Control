"""
Driver function that defines the system

--- Takes the following nonlinear problem ---
Consider a simple aerodynamic problem in which a point mass is traveling through 2D space. To accelerate, this
mass has thrusters, ux and uy (that act in their respective directions). Furthermore, this mass encounters a simple
drag force, which it must overcome to reach the desired point. This problem is modeled by:

x = [px, py, vx, vy] --> State Vector
u = [ux, uy] --> Control Vector
\dot{px} = vx
\dot{py} = vy
\dot{vx} = ux - c * vx * sqrt(vx^2 + vy^2)
\dot{vy} = uy - c * vy * sqrt(vx^2 + vy^2)

where c is the drag constant (e.g. 0.4).

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

# Define linearization:
def linearization(vx, vy, c):

    # Define derivative terms for A:
    sqrt_term = np.sqrt(vx**2 + vy**2)
    if sqrt_term == 0.0:
        sqrt_term = 1e-2 # Pad from divide by zero
    dvx_dvx = -c * (sqrt_term + vx**2 / sqrt_term)
    dvx_dvy = -c * vx * vy / sqrt_term
    dvy_dvx = -c * vy * vx / sqrt_term
    dvy_dvy = -c * (sqrt_term + vy**2 / sqrt_term)

    # Construct A matrix:
    a_mat = np.array([[0, 0,     1,      0   ],
                      [0, 0,     0,      1   ],
                      [0, 0, dvx_dvx, dvx_dvy],
                      [0, 0, dvy_dvx, dvy_dvy]])

    # Construct B matrix:
    b_mat = np.array([[0, 0],
                      [0, 0],
                      [1, 0],
                      [0, 1]])

    # Construct C matrix (just identity matrix):
    c_mat = np.eye(4)

    # Return A, B, and C matrices:
    return a_mat, b_mat, c_mat

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

# Problem parameters:
c = 0.4 # Drag constant

# Initial conditions:
x0 = np.array([0, 0, 0, 0])

# Define weight scales:
Q0_scaler = 1e-3
Q_scaler = 1e-4
P_scaler = 1e4

# Define control bounds:
u_max = np.array([20, 20]).T
u_min = np.array([-10, -10]).T

####################################################################

# --- Define weight matrices:
# Penalize current size of u:
Q0 = 0.00001 * np.eye(2)
Q0 *= Q0_scaler

# Penalize difference between u values at subsequent steps:
Q = 0.0001 * np.eye(2)
Q *= Q_scaler

# Penalize error in objective:
P = 10 * np.eye(4)
P *= P_scaler
P[2, 2] = 0
P[3, 3] = 0  # Don't really care about velocities, so don't penalize

# --- Develop MPC problem ---
# Fetch dimensions:
r = 4
m = Q0.shape[0]
n = P.shape[0]

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
traj = np.ones((n_tsteps, 4))
traj[:, 0] *= 10
traj[:, 1] *= 5
traj[:, 2] *= 0
traj[:, 3] *= 0

# --- Use MPC controller ---
# Build MPC object:
mpc = MPC(None, None, None, f, v, W3, W4, x0, traj, u_min, u_max, 'nonlinear')

# Initialize velocities from initial condition (need for linearization):
vx = x0[2]
vy = x0[3]

# Use controller:
for i in range(n_tsteps - f):

    # Linearize for new A, B, and C (continuous) matrices:
    A_cts, B_cts, C_cts = linearization(vx, vy, c)

    # Discretize matrices for MPC:
    A, B = discretize_system(A_cts, B_cts, dt)
    C = C_cts

    # Run MPC step:
    mpc.control_inputs(A, B, C)

    # Fetch outputs (will be used in jacobian calculation):
    vx = mpc.outputs[-1][2, 0]
    vy = mpc.outputs[-1][3, 0]

# --- Unpack, plot ---
# Extract state estimates:
y_des = []
y = []
ctrl = []
for j in np.arange(n_tsteps - f):
    y.append(mpc.outputs[j][:, 0])
    y_des.append(traj[j, :])
    ctrl.append(mpc.inputs[j][:])

# Switch to arrays:
y = np.array(y)
y_des = np.array(y_des)
ctrl = np.array(ctrl)

# Plot graphs:
plot_graphs(time_ctrl, y, y_des, ctrl, 'NL_Drag', np.array([u_min, u_max]))

# Show results:
plt.show()
