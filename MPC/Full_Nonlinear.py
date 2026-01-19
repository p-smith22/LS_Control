# Import packages:
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time

# === FUNCTIONS ===
# Compute state derivatives for CasADi:
def f_nl(x, u, c):

    # Compute state derivatives and return:
    px, py, vx, vy = x[0], x[1], x[2], x[3]
    ux, uy = u[0], u[1]
    sqrt_term = ca.sqrt(vx**2 + vy**2)
    sqrt_term += 1e-4
    return ca.vertcat(vx, vy, ux - c * vx * sqrt_term, uy - c * vy * sqrt_term)

# Take a TRUE nonlinear step using RK4:
def nonlinear_step(x, u, dt, c):

    # Time-derivative function:
    def f(x, u):
        px, py, vx, vy = x
        ux, uy = u
        v = np.sqrt(vx**2 + vy**2 + 1e-6)
        return np.array([vx, vy, ux - c * vx * v, uy - c * vy * v])

    # Propagate dynamics and return:
    k1 = f(x, u)
    k2 = f(x + dt/2 * k1, u)
    k3 = f(x + dt/2 * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# Linearize around a given reference point (defined by velocity):
def linearization(vx, vy, c):

    # Precompute square root term (adds epsilon term to avoid divide by zero):
    sqrt_term = np.sqrt(vx**2 + vy**2)
    sqrt_term += 1e-4

    # Compute jacobian:
    dvx_dvx = -c * (sqrt_term + vx**2 / sqrt_term)
    dvx_dvy = -c * vx * vy / sqrt_term
    dvy_dvx = -c * vy * vx / sqrt_term
    dvy_dvy = -c * (sqrt_term + vy**2 / sqrt_term)

    # Assemble matrices and return:
    A = np.array([[0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, dvx_dvx, dvx_dvy],
                  [0, 0, dvy_dvx, dvy_dvy]])
    B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
    C = np.eye(4)
    return A, B, C

# Build nonlinear MPC problem:
def build_nlmpc(c, dt, N, v, C_mat, nx=4, nu=2, ny=4, umin=None, umax=None, W3=None, W4=None):

    # Define symbols for problem:
    x = ca.SX.sym('x', nx)
    u = ca.SX.sym('u', nu)
    du = ca.SX.sym('DU', nu, v)

    # RK4 integration for propagation:
    k1 = f_nl(x, u, c)
    k2 = f_nl(x + dt/2 * k1, u, c)
    k3 = f_nl(x + dt/2 * k2, u, c)
    k4 = f_nl(x + dt * k3, u, c)
    x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    f_disc = ca.Function('f_disc', [x, u], [x_next])

    # Parameters:
    x0_param = ca.SX.sym('X0', nx)
    yref_param = ca.SX.sym('Yref', ny, N)

    # Build controls:
    u_vals = ca.SX.zeros(nu, N)
    for i in range(N):
        for j in range(min(i+1, v)):
            u_vals[:, i] += du[:, j]

    # Simulate dynamics:
    x_vals = ca.SX.zeros(nx, N+1)
    x_vals[:, 0] = x0_param
    for i in range(N):
        x_vals[:, i+1] = f_disc(x_vals[:, i], u_vals[:, i])

    # Extract outputs:
    y_vals = ca.SX.zeros(ny, N)
    for i in range(N):
        y_vals[:, i] = ca.mtimes(C_mat, x_vals[:, i+1])

    # Reshape:
    y_vals = ca.reshape(y_vals, ny*N, 1)
    yref_param = ca.reshape(yref_param, ny*N, 1)
    du = ca.reshape(du, nu*v, 1)

    # Calculate cost:
    e_y = y_vals - yref_param
    J = ca.mtimes([e_y.T, W4, e_y]) + ca.mtimes([du.T, W3, du])

    # Control bounds:
    U_vec = ca.reshape(u_vals, nu*N, 1)

    # NLP formulation:
    opt_vars = du
    params = ca.vertcat(x0_param, yref_param)
    nlp = {
        'x': opt_vars,
        'f': J,
        'g': U_vec,
        'p': params
    }
    opts = {
        'ipopt.print_level': 0,
        'print_time': False,
        'ipopt.max_iter': 500,
        'ipopt.tol': 1e-6,
        'ipopt.acceptable_tol': 1e-4,
        'ipopt.warm_start_init_point': 'yes',
        'ipopt.mu_strategy': 'adaptive'
    }

    # Set solver:
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Bounds:
    lbx = [-ca.inf] * (nu*v)
    ubx = [ca.inf] * (nu*v)
    lbg = np.tile(umin, N)
    ubg = np.tile(umax, N)

    # Return built problem:
    return solver, lbx, ubx, lbg, ubg

# Solve the constructed MPC problem:
def solver(solver, lbx, ubx, lbg, ubg, x0, yref, nu, v, prev_du=None):

    # Warm start (shift previous step to make it faster):
    if prev_du is None:
        init_guess = np.zeros(nu*v)
    else:
        du_prev = prev_du.reshape((v, nu))
        du_new = np.vstack([du_prev[1:], np.zeros((1, nu))])
        init_guess = du_new.flatten()

    # Build parameter vector:
    params = np.concatenate([x0, yref.flatten(order='F')])

    # Solve problem:
    sol = solver(x0=init_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=params)
    du_opt = sol['x'].full().flatten()

    # Take first control:
    u0 = du_opt[:nu]

    # Return:
    return u0, du_opt

# === SETUP COMPARISON ===
# Parameters:
c = 2.0
dt = 0.01
t_end = 60.0
n_tsteps = int(t_end / dt)

# MPC parameters:
f = 50
v = 20

# Initial condition:
x0 = np.array([0.0, 0.0, 2.0, 0.0])

# Control bounds:
u_min = np.array([-50.0, -50.0])
u_max = np.array([50.0, 50.0])

# Generate reference trajectory:
amplitude = 25.0
omega = 2 * np.pi / 40.0
traj = np.zeros((n_tsteps, 4))
for i in range(n_tsteps):
    t = i * dt
    traj[i, 0] = x0[2] * t
    traj[i, 1] = amplitude * np.sin(omega * t)
    traj[i, 2] = x0[2]
    traj[i, 3] = amplitude * omega * np.cos(omega * t)

# Weight matrices:
Q0 = 0.001 * np.eye(2)
Q = 0.1 * np.eye(2)
P_full = np.diag([10000, 10000, 10000, 10000])

# Output matrix (perfectly observable):
C = np.eye(4)
r = C.shape[0]
m = 2

# Build W1, W2, W3, W4:
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

# Initialize:
n_sim = n_tsteps - f
results = {}

# === NONLINEAR MPC ===
# Transfer to CasADi variables:
W3_ca = ca.DM(W3)
W4_ca = ca.DM(W4)
C_ca = ca.DM(C)

# Build nonlinear MPC problem:
solver_nl, lbx, ubx, lbg, ubg = build_nlmpc(
    c, dt, f, v, C_ca, nx=4, nu=2, ny=4,
    umin=u_min, umax=u_max, W3=W3_ca, W4=W4_ca
)

# Initialize variables:
x_nl = np.zeros((n_sim, 4))
u_nl = np.zeros((n_sim, 2))
x_current = x0.copy()
prev_du = None

# Solve nonlinear MPC problem:
start = time.perf_counter()
for k in range(n_sim):

    # Reference outputs:
    yref = np.zeros((r, f))
    for i in range(f):
        yref[:, i] = C @ traj[k+1+i, :]

    # Solve time step:
    u_opt, prev_du = solver(
        solver_nl, lbx, ubx, lbg, ubg, x_current, yref, 2, v, prev_du
    )

    # Assign time step:
    x_current = nonlinear_step(x_current, u_opt, dt, c)
    x_nl[k, :] = x_current
    u_nl[k, :] = u_opt

# Calculate values and store in results:
time_nl = time.perf_counter() - start
error_nl = np.sum((x_nl[:, :2] - traj[:n_sim, :2])**2)
cost_nl = np.sum(u_nl**2)
results['nonlinear'] = {'x': x_nl, 'u': u_nl, 'time': time_nl, 'error': error_nl, 'cost': cost_nl}

# Calculate instantaneous results (for plotting):
cost_nl_inst = np.cumsum(np.sum(u_nl**2, axis=1))

# Calculate positional tracking error (for plotting):
err_nl = np.linalg.norm(x_nl[:, :2] - traj[:n_sim, :2], axis=1)

# === PRINT RESULTS ===
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"{'Method':<15} {'Runtime (s)':<15} {'Traj Error':<15} {'Control Cost':<15}")
print("-"*60)
print(f"{'Nonlinear':<15} {time_nl:<15.3f} {error_nl:<15.2f} {cost_nl:<15.2f}")
print("="*60)

# === PLOT RESULTS ===
# Compute time vector for plotting:
time_vec = np.arange(n_sim) * dt

# Create figure:
fig, axes = plt.subplots(4, 2, figsize=(14, 10))
fig.suptitle('Fully Nonlinear MPC', fontsize=16, fontweight='bold')

# Position (px):
axes[0, 0].plot(time_vec, traj[:n_sim, 0], 'k--', linewidth=2, label='Reference')
axes[0, 0].plot(time_vec, x_nl[:, 0], 'b-', label='Nonlinear', alpha=0.8)
axes[0, 0].set_ylabel('px (m)', fontsize=11)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(loc='best')

# Position (py):
axes[1, 0].plot(time_vec, traj[:n_sim, 1], 'k--', linewidth=2, label='Reference')
axes[1, 0].plot(time_vec, x_nl[:, 1], 'b-', label='Nonlinear', alpha=0.8)
axes[1, 0].set_ylabel('py (m)', fontsize=11)
axes[1, 0].grid(True, alpha=0.3)

# Velocity (vx):
axes[2, 0].plot(time_vec, traj[:n_sim, 2], 'k--', linewidth=2, label='Reference')
axes[2, 0].plot(time_vec, x_nl[:, 2], 'b-', label='Nonlinear', alpha=0.8)
axes[2, 0].set_ylabel('vx (m/s)', fontsize=11)
axes[2, 0].grid(True, alpha=0.3)

# Velocity (vy):
axes[3, 0].plot(time_vec, traj[:n_sim, 3], 'k--', linewidth=2, label='Reference')
axes[3, 0].plot(time_vec, x_nl[:, 3], 'b-', label='Nonlinear', alpha=0.8)
axes[3, 0].set_ylabel('vy (m/s)', fontsize=11)
axes[3, 0].set_xlabel('Time (s)', fontsize=11)
axes[3, 0].grid(True, alpha=0.3)

# Control (ux):
axes[0, 1].plot(time_vec, u_nl[:, 0], 'b-', label='Nonlinear', alpha=0.8)
axes[0, 1].axhline(u_max[0], color='k', linestyle=':', alpha=0.5)
axes[0, 1].axhline(u_min[0], color='k', linestyle=':', alpha=0.5)
axes[0, 1].set_ylabel('ux (N)', fontsize=11)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend(loc='best')

# Control (uy):
axes[1, 1].plot(time_vec, u_nl[:, 1], 'b-', label='Nonlinear', alpha=0.8)
axes[1, 1].axhline(u_max[1], color='k', linestyle=':', alpha=0.5)
axes[1, 1].axhline(u_min[1], color='k', linestyle=':', alpha=0.5)
axes[1, 1].set_ylabel('uy (N)', fontsize=11)
axes[1, 1].grid(True, alpha=0.3)

# Tracking error:
axes[2, 1].plot(time_vec, err_nl, 'b-', label='Nonlinear', alpha=0.8)
axes[2, 1].set_ylabel('Tracking Error (m)', fontsize=11)
axes[2, 1].set_yscale('log')
axes[2, 1].grid(True, alpha=0.3)
axes[2, 1].legend(loc='best')

# Control cost:
axes[3, 1].plot(time_vec, cost_nl_inst, 'b-', label='Nonlinear', alpha=0.8)
axes[3, 1].set_xlabel('Time (s)', fontsize=11)
axes[3, 1].set_ylabel('Cumulative Cost $\\|u_k\\|^2$', fontsize=11)
axes[3, 1].set_yscale('log')
axes[3, 1].grid(True, alpha=0.3)
axes[3, 1].legend(loc='best')

# Plot settings:
plt.tight_layout()
plt.savefig('./output/mpc_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
