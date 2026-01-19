# Import packages:
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time
from src.MPC import MPC
from src.utils import discretize_system
from src.nlmpc import build_nlmpc, solver

# === SELECT METHODS TO RUN ===
# Options = 'nonlinear', 'ltv', and 'lti'
methods_to_run = ['ltv']

# === FUNCTIONS ===
def nonlinear_step(x, u, dt, c):
    def f(x, u):
        px, py, vx, vy = x
        ux, uy = u
        v = np.sqrt(vx**2 + vy**2 + 1e-6)
        return np.array([vx, vy, ux - c * vx * v, uy - c * vy * v])
    k1 = f(x, u)
    k2 = f(x + dt/2 * k1, u)
    k3 = f(x + dt/2 * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def linearization(vx, vy, c):
    sqrt_term = np.sqrt(vx**2 + vy**2) + 1e-4
    dvx_dvx = -c * (sqrt_term + vx**2 / sqrt_term)
    dvx_dvy = -c * vx * vy / sqrt_term
    dvy_dvx = -c * vy * vx / sqrt_term
    dvy_dvy = -c * (sqrt_term + vy**2 / sqrt_term)
    A = np.array([[0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, dvx_dvx, dvx_dvy],
                  [0, 0, dvy_dvx, dvy_dvy]])
    B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
    C = np.eye(4)
    return A, B, C

def plot_traj(ax, ref, *data, labels):
    ax.plot(time_vec, ref, 'k--', linewidth=2, label='Reference')
    for d, l in zip(data, labels):
        ax.plot(time_vec, d, label=l, alpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

def plot_control(ax, *data, labels, umin, umax):
    for d, l in zip(data, labels):
        ax.plot(time_vec, d, label=l, alpha=0.8)
    ax.axhline(umax, color='k', linestyle=':', alpha=0.5)
    ax.axhline(umin, color='k', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

# === SETUP SIMULATION ===
# Parameters:
c = 2.0
dt = 0.01
t_end = 60.0
n_tsteps = int(t_end / dt)
f = 100
v = 20
x0 = np.array([0.0, 0.0, 2.0, 0.0])
u_min = np.array([-50.0, -50.0])
u_max = np.array([50.0, 50.0])

# Reference trajectory:
amplitude = 25.0
omega = 2 * np.pi / 40.0
traj = np.zeros((n_tsteps, 4))
for i in range(n_tsteps):
    t = i * dt
    traj[i, 0] = x0[2] * t
    traj[i, 1] = amplitude * np.sin(omega * t)
    traj[i, 2] = x0[2]
    traj[i, 3] = amplitude * omega * np.cos(omega * t)

# Weights:
Q0 = 0.01 * np.eye(2)
Q = 10 * np.eye(2)
P_full = np.diag([1000, 1000, 100, 100])
C = np.eye(4)
r = C.shape[0]
m = 2

# Build W3, W4:
W1 = np.zeros((v*m, v*m))
for i in range(v):
    W1[i*m:(i+1)*m, i*m:(i+1)*m] = np.eye(m)
    if i > 0:
        W1[i*m:(i+1)*m, (i-1)*m:i*m] = -np.eye(m)
W2 = np.zeros((v*m, v*m))
for i in range(v):
    W2[i*m:(i+1)*m, i*m:(i+1)*m] = Q0 if i == 0 else Q
W3 = W1.T @ W2 @ W1
W4 = np.kron(np.eye(f), P_full)

# Number of simulation steps:
n_sim = n_tsteps - f
results = {}

# === NONLINEAR MPC ===
if 'nonlinear' in methods_to_run:

    # Make variables:
    W3_ca = ca.DM(W3)
    W4_ca = ca.DM(W4)
    C_ca = ca.DM(C)

    # Build MPC problem:
    solver_nl, lbx, ubx, lbg, ubg = build_nlmpc(c, dt, f, v, C_ca, nx=4, nu=2, ny=4,
                                                umin=u_min, umax=u_max, W3=W3_ca, W4=W4_ca)

    # Initialize variables:
    x_nl = np.zeros((n_sim, 4))
    u_nl = np.zeros((n_sim, 2))
    x_current = x0.copy()
    prev_du = None

    # Solve the system:
    start = time.perf_counter()
    for k in range(n_sim):

        # Set reference output:
        yref = np.zeros((r, f))
        for i in range(f):
            yref[:, i] = C @ traj[k+1+i, :]

        # Solve MPC:
        u_opt, prev_du = solver(solver_nl, lbx, ubx, lbg, ubg, x_current, yref, 2, v, prev_du)

        # Propagate dynamics:
        x_current = nonlinear_step(x_current, u_opt, dt, c)

        # Assign current values:
        x_nl[k, :] = x_current
        u_nl[k, :] = u_opt

    # Calculate results:
    time_nl = time.perf_counter() - start
    error_nl = np.sum((x_nl[:, :2] - traj[:n_sim, :2]) ** 2)
    cost_nl = np.sum(u_nl**2)
    results['nonlinear'] = {'x': x_nl, 'u': u_nl, 'time': time_nl, 'error': error_nl, 'cost': cost_nl}

# === LTV MPC ===
if 'ltv' in methods_to_run:

    # Build MPC object:
    mpc_ltv = MPC(None, None, None, f, v, W3, W4, x0, traj, u_min, u_max, 'nonlinear', 'LTV')

    # Initialize states:
    x_ltv = np.zeros((n_tsteps - f, 4))
    u_ltv = np.zeros((n_tsteps - f, 2))
    x_k = x0.copy()

    # Start timer:
    start = time.perf_counter()

    # Use controller:
    for i in range(n_tsteps - f):

        # Initialize sequential matrices:
        A_seq = []
        B_seq = []
        C_seq = []

        # Predict trajectory forward using current state:
        x_pred = x_k.copy()

        for j in range(f):

            # Linearize at predicted state:
            vx_pred, vy_pred = x_pred[2], x_pred[3]
            A_cts, B_cts, C_cts = linearization(vx_pred, vy_pred, c)

            # Discretize:
            A_disc, B_disc = discretize_system(A_cts, B_cts, dt)

            # Store in sequences:
            A_seq.append(A_disc)
            B_seq.append(B_disc)
            C_seq.append(C_cts)

            # Grab last control:
            if len(mpc_ltv.inputs) > 0 and j == 0:
                u_nom = mpc_ltv.inputs[-1]
            else:
                u_nom = np.zeros(2)

            # Propagate prediction:
            x_pred = nonlinear_step(x_pred, u_nom, dt, c)

        # Pass sequence:
        mpc_ltv.control_inputs(A_seq, B_seq, C_seq)

        # Fetch last control input:
        u_k = mpc_ltv.inputs[-1]

        # Propagate nonlinear dynamics:
        x_k = nonlinear_step(x_k, u_k, dt, c)

        # Feed true state back to MPC:
        mpc_ltv.states.append(x_k.reshape(-1, 1))

        # Store current time step:
        x_ltv[i, :] = x_k.reshape(-1)
        u_ltv[i, :] = u_k.flatten()

    # Calculate results:
    time_ltv = time.perf_counter() - start
    error_ltv = np.sum((x_ltv[:, :2] - traj[:n_sim, :2]) ** 2)
    cost_ltv = np.sum(u_ltv ** 2)
    results['ltv'] = {'x': x_ltv, 'u': u_ltv, 'time': time_ltv, 'error': error_ltv, 'cost': cost_ltv}

# === LTI MPC ===
if 'lti' in methods_to_run:

    # Linearize and discretize trim matrices:
    A_cts, B_cts, C_cts = linearization(x0[2], x0[3], c)
    A_lti, B_lti = discretize_system(A_cts, B_cts, dt)

    # Initialize MPC object:
    mpc_lti = MPC(None, None, None, f, v, W3, W4, x0, traj, u_min, u_max, 'nonlinear', 'LTI')

    # Initialize variables:
    x_lti = np.zeros((n_sim, 4))
    u_lti = np.zeros((n_sim, 2))
    x_current = x0.copy()

    # Solve MPC problem:
    start = time.perf_counter()
    for k in range(n_sim):

        # Solve:
        mpc_lti.control_inputs(A_lti, B_lti, C_cts)

        # Extract control:
        u_k = mpc_lti.inputs[-1]

        # Propagate dynamics:
        x_current = nonlinear_step(x_current, u_k, dt, c)

        # Extract states:
        mpc_lti.states.append(x_current.reshape(-1, 1))

        # Assign current step:
        x_lti[k, :] = x_current
        u_lti[k, :] = u_k.flatten()

    # Calculate results:
    time_lti = time.perf_counter() - start
    error_lti = np.sum((x_lti[:, :2] - traj[:n_sim, :2]) ** 2)
    cost_lti = np.sum(u_lti**2)
    results['lti'] = {'x': x_lti, 'u': u_lti, 'time': time_lti, 'error': error_lti, 'cost': cost_lti}

# === PRINT SUMMARY ===
methods_present = [m for m in ['nonlinear', 'ltv', 'lti'] if m in results]
print("\n" + "="*60)
print("SIMULATION RESULTS")
print("="*60)
print(f"{'Method':<15} {'Runtime (s)':<15} {'Traj Error':<15} {'Control Cost':<15}")
print("-"*60)
for m in methods_present:
    print(f"{m:<15} {results[m]['time']:<15.3f} {results[m]['error']:<15.2f} {results[m]['cost']:<15.2f}")
print("="*60)

# === PLOTTING ===
time_vec = np.arange(n_sim) * dt
fig, axes = plt.subplots(4, 2, figsize=(14, 10))
fig.suptitle('MPC Comparison', fontsize=16, fontweight='bold')

# Position px:
plot_traj(axes[0, 0], traj[:n_sim, 0],
          *[results[m]['x'][:, 0] for m in methods_present], labels=methods_present)
axes[0, 0].set_ylabel('px (m)')

# Position py:
plot_traj(axes[1, 0], traj[:n_sim, 1],
          *[results[m]['x'][:, 1] for m in methods_present], labels=methods_present)
axes[1, 0].set_ylabel('py (m)')

# Velocity vx:
plot_traj(axes[2, 0], traj[:n_sim, 2],
          *[results[m]['x'][:, 2] for m in methods_present], labels=methods_present)
axes[2, 0].set_ylabel('vx (m/s)')

# Velocity vy:
plot_traj(axes[3, 0], traj[:n_sim, 3],
          *[results[m]['x'][:, 3] for m in methods_present], labels=methods_present)
axes[3, 0].set_ylabel('vy (m/s)')
axes[3, 0].set_xlabel('Time (s)')

# Control ux:
plot_control(axes[0, 1],
             *[results[m]['u'][:, 0] for m in methods_present], labels=methods_present,
             umin=u_min[0], umax=u_max[0])
axes[0, 1].set_ylabel('ux (N)')

# Control uy:
plot_control(axes[1, 1],
             *[results[m]['u'][:, 1] for m in methods_present], labels=methods_present,
             umin=u_min[1], umax=u_max[1])
axes[1, 1].set_ylabel('uy (N)')

# Tracking error:
for m in methods_present:
    err = np.linalg.norm(results[m]['x'][:, :2] - traj[:n_sim, :2], axis=1)
    axes[2, 1].plot(time_vec, err, label=m, alpha=0.8)
axes[2, 1].set_ylabel('Tracking Error (m)')
axes[2, 1].set_yscale('log')
axes[2, 1].grid(True, alpha=0.3)
axes[2, 1].legend(loc='best')

# Control cost:
for m in methods_present:
    cost_inst = np.cumsum(np.sum(results[m]['u']**2, axis=1))
    axes[3, 1].plot(time_vec, cost_inst, label=m, alpha=0.8)
axes[3, 1].set_xlabel('Time (s)')
axes[3, 1].set_ylabel('Control Cost (Cum.)')
axes[3, 1].set_yscale('log')
axes[3, 1].grid(True, alpha=0.3)
axes[3, 1].legend(loc='best')

# Plot:
plt.tight_layout()
plt.show()
