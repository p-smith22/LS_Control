# Import packages:
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time
from src.MPC import MPC
from src.nlmpc import build_nlmpc, solver

# === SELECT METHODS TO RUN ===
# Options = 'nonlinear', 'ltv', and 'lti'
methods_to_run = ['lti', 'ltv', 'nonlinear']

# === FUNCTIONS ===
# Function to take a nonlinear time step:
def nonlinear_step(x, u, dt, c):

    # Helper function that contains the time derivatives:
    def f(x, u):
        px, py, vx, vy = x
        ux, uy = u
        sqrt_term = np.sqrt(vx ** 2 + vy ** 2)
        return np.array([vx, vy, ux - c * vx * sqrt_term, uy - c * vy * sqrt_term])

    # Propagate using RK4:
    k1 = f(x, u)
    k2 = f(x + dt / 2 * k1, u)
    k3 = f(x + dt / 2 * k2, u)
    k4 = f(x + dt * k3, u)

    # Return next time step:
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# Function that linearizes to obtain (continuous) matrices:
def lin_cts(x, u, c):

    # Unpack variables:
    px, py, vx, vy = x
    ux, uy = u

    # Pre-compute square root term:
    sqrt_term = np.sqrt(vx ** 2 + vy ** 2) + 1e-6

    # Calculate jacobian:
    dvx_dvx = -c * (sqrt_term + vx ** 2 / sqrt_term)
    dvx_dvy = -c * vx * vy / sqrt_term
    dvy_dvx = -c * vy * vx / sqrt_term
    dvy_dvy = -c * (sqrt_term + vy ** 2 / sqrt_term)

    # Assemble matrices:
    A_c = np.array([[0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, dvx_dvx, dvx_dvy],
                    [0, 0, dvy_dvx, dvy_dvy]])
    B_c = np.array([[0, 0],
                    [0, 0],
                    [1, 0],
                    [0, 1]])
    C = np.eye(4)

    # Compute affine term g_c = f(x,u) - (A_c*x + B_c*u):
    f_xu = np.array([vx, vy, ux - c * vx * sqrt_term, uy - c * vy * sqrt_term])
    g_c = f_xu - A_c @ x - B_c @ u

    # Return continuous matrices and affine term:
    return A_c, B_c, C, g_c

# Discretize system to get x_(k+1) = A_d*x + B_d*x + g_d:
def disc_sys(A_c, B_c, g_c, dt):

    # Fetch dimensions:
    n = A_c.shape[0]
    m = B_c.shape[1]

    # Assemble augmented matrix:
    M = np.zeros((n + m + 1, n + m + 1))
    M[:n, :n] = A_c
    M[:n, n:n + m] = B_c
    M[:n, n + m] = g_c

    # Matrix exponential (Taylor Approx.):
    M_d = np.eye(n + m + 1) + M * dt + (M @ M) * (dt ** 2 / 2)

    # Extract discrete matrices:
    A_d = M_d[:n, :n]
    B_d = M_d[:n, n:n + m]
    g_d = M_d[:n, n + m]

    # Return discretized matrices and affine term:
    return A_d, B_d, g_d

# Compute reference control sequence for a given state sequence:
def compute_uref(x_ref_seq, dt, c):

    # Initialize list:
    u_ref_seq = []

    # Loop through each of the reference states in the sequence:
    for k in range(len(x_ref_seq) - 1):

        # Calculate required accelerations given change in velocity:
        ax_req = (x_ref_seq[k+1][2] - x_ref_seq[k][2]) / dt
        ay_req = (x_ref_seq[k+1][3] - x_ref_seq[k][3]) / dt

        # Calculate control from current velocity and required acceleration:
        sqrt_term = np.sqrt(x_ref_seq[k][2] ** 2 + x_ref_seq[k][3] ** 2)
        ux_ref = ax_req + c * x_ref_seq[k][2] * sqrt_term
        uy_ref = ay_req + c * x_ref_seq[k][3] * sqrt_term

        # Append reference controls to sequence:
        u_ref_seq.append(np.array([ux_ref, uy_ref]))

    # Return reference control sequence (for all the steps):
    return u_ref_seq

# Plot states (trajectory):
def plot_traj(ax, ref, *data, labels):
    ax.plot(time_vec, ref, 'k--', linewidth=2, label='Reference')
    for d, l in zip(data, labels):
        ax.plot(time_vec, d, label=l, alpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

# Plot controls:
def plot_control(ax, *data, labels, umin, umax):
    for d, l in zip(data, labels):
        ax.plot(time_vec, d, label=l, alpha=0.8)
    ax.axhline(umax, color='k', linestyle=':', alpha=0.5)
    ax.axhline(umin, color='k', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

# === SETUP SIMULATION ===
# ------ IMPORTANT TUNING PARAMETERS ------
# Horizon variables:
f = 50
v = 20

# Weights:
Q0 = 0.001 * np.eye(2)
Q = 0.1 * np.eye(2)
P_full = np.diag([10000, 10000, 10000, 10000])
# -----------------------------------------

# Other parameters:
c = 2.0
dt = 0.01
t_end = 60.0
n_tsteps = int(t_end / dt)
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

# Assign observability (fully observable = I):
C = np.eye(4)

# Get dimensions:
r = C.shape[0]
m = 2

# Build W3, W4:
W1 = np.zeros((v * m, v * m))
for i in range(v):
    W1[i * m:(i + 1) * m, i * m:(i + 1) * m] = np.eye(m)
    if i > 0:
        W1[i * m:(i + 1) * m, (i - 1) * m:i * m] = -np.eye(m)
W2 = np.zeros((v * m, v * m))
for i in range(v):
    W2[i * m:(i + 1) * m, i * m:(i + 1) * m] = Q0 if i == 0 else Q
W3 = W1.T @ W2 @ W1
W4 = np.kron(np.eye(f), P_full)

# Number of simulation steps:
n_sim = n_tsteps - f
results = {}

# === LTI MPC ===
if 'lti' in methods_to_run:

    # Linearize and discretize at trim point
    A_c, B_c, C_c, g_c = lin_cts(x0, np.zeros(2), c)
    A_lti, B_lti, g_lti = disc_sys(A_c, B_c, g_c, dt)

    # Note: For LTI around a trim point, g should be zero if the trim satisfies f(x0,u0)=0
    # Here it won't be zero because x0 has non-zero velocity

    # Initialize MPC object:
    mpc_lti = MPC(None, None, None, f, v, W3, W4, x0, traj, u_min, u_max, 'nonlinear', 'LTI')

    # Initialize variables:
    x_lti = np.zeros((n_sim, 4))
    u_lti = np.zeros((n_sim, 2))
    x_current = x0.copy()

    # Solve MPC problem:
    start = time.perf_counter()
    for k in range(n_sim):
        # Solve (ignoring affine term for LTI - could be added but typically negligible)
        mpc_lti.control_inputs(A_lti, B_lti, C_c)

        # Extract control:
        u_k = mpc_lti.inputs[-1].flatten()

        # Propagate true nonlinear dynamics:
        x_current = nonlinear_step(x_current, u_k, dt, c)

        # Extract states:
        mpc_lti.states.append(x_current.reshape(-1, 1))

        # Assign current step:
        x_lti[k, :] = x_current
        u_lti[k, :] = u_k

    # Calculate results:
    time_lti = time.perf_counter() - start
    error_lti = np.sum((x_lti[:, :2] - traj[:n_sim, :2]) ** 2)
    cost_lti = np.sum(u_lti ** 2)
    results['lti'] = {'x': x_lti, 'u': u_lti, 'time': time_lti, 'error': error_lti, 'cost': cost_lti}

# === LTV MPC ===
if 'ltv' in methods_to_run:

    # Build MPC object:
    mpc_ltv = MPC(None, None, None, f, v, W3, W4, x0, traj, u_min, u_max, 'nonlinear', 'LTV')

    # Initialize states:
    x_ltv = np.zeros((n_sim, 4))
    u_ltv = np.zeros((n_sim, 2))
    x_k = x0.copy()

    # Start timer:
    start = time.perf_counter()

    # Use controller:
    for i in range(n_sim):

        # Extract reference trajectory:
        x_ref_seq = [traj[i + j].copy() for j in range(f + 1)]
        u_ref_seq = compute_uref(x_ref_seq, dt, c)

        # Initialize sequential matrices:
        A_seq = []
        B_seq = []
        C_seq = []
        d_seq = []

        # Linearize about reference trajectory:
        for j in range(f):

            # Get linearized matrices:
            A_c, B_c, C_c, g_c = lin_cts(x_ref_seq[j], u_ref_seq[j], c)

            # Discretize:
            A_d = np.eye(4) + A_c * dt
            B_d = B_c * dt

            # Compute affine offset variable:
            d_j = x_ref_seq[j + 1] - A_d @ x_ref_seq[j] - B_d @ u_ref_seq[j]

            # Store in sequences:
            A_seq.append(A_d)
            B_seq.append(B_d)
            C_seq.append(C_c)
            d_seq.append(d_j)

        # Solve MPC:
        mpc_ltv.control_inputs(A_seq, B_seq, C_seq, x_ref_seq, u_ref_seq, d_seq)

        # Extract last control:
        u_k = mpc_ltv.inputs[-1].flatten()

        # Propagate TRUE nonlinear dynamics:
        x_k = nonlinear_step(x_k, u_k, dt, c)

        # Feed true state back to MPC:
        mpc_ltv.states.append(x_k.reshape(-1, 1))

        # Store current time step:
        x_ltv[i, :] = x_k
        u_ltv[i, :] = u_k

    # Calculate results:
    time_ltv = time.perf_counter() - start
    error_ltv = np.sum((x_ltv[:, :2] - traj[:n_sim, :2]) ** 2)
    cost_ltv = np.sum(u_ltv ** 2)
    results['ltv'] = {'x': x_ltv, 'u': u_ltv, 'time': time_ltv, 'error': error_ltv, 'cost': cost_ltv}

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
            yref[:, i] = C @ traj[k + 1 + i, :]

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
    cost_nl = np.sum(u_nl ** 2)
    results['nonlinear'] = {'x': x_nl, 'u': u_nl, 'time': time_nl, 'error': error_nl, 'cost': cost_nl}

# === PRINT SUMMARY ===
methods_present = [m for m in ['nonlinear', 'ltv', 'lti'] if m in results]
print("\n" + "=" * 60)
print("SIMULATION RESULTS (LTV Linearized Around Reference)")
print("=" * 60)
print(f"{'Method':<15} {'Runtime (s)':<15} {'Traj Error':<15} {'Control Cost':<15}")
print("-" * 60)
for m in methods_present:
    print(f"{m:<15} {results[m]['time']:<15.3f} {results[m]['error']:<15.2f} {results[m]['cost']:<15.2f}")
print("=" * 60)

# === PLOTTING ===
time_vec = np.arange(n_sim) * dt
fig, axes = plt.subplots(4, 2, figsize=(14, 10))
fig.suptitle('MPC Comparison (LTV Around Reference)', fontsize=16, fontweight='bold')

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
    cost_inst = np.cumsum(np.sum(results[m]['u'] ** 2, axis=1))
    axes[3, 1].plot(time_vec, cost_inst, label=m, alpha=0.8)
axes[3, 1].set_xlabel('Time (s)')
axes[3, 1].set_ylabel('Control Cost (Cum.)')
axes[3, 1].set_yscale('log')
axes[3, 1].grid(True, alpha=0.3)
axes[3, 1].legend(loc='best')

# Plot:
plt.tight_layout()
plt.show()