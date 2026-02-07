# Import packages:
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time
from src.MPC import MPC
from src.nlmpc import build_nlmpc, solver

# === SELECT METHODS TO RUN ===
# Options = 'LTI', 'LTV', and/or 'NL'
methods_to_run = ['LTI', 'LTV', 'NL']

# === MONTE CARLO CONFIGURATION ===
run_monte_carlo = True
n_trials = 5
# Noise standard deviation [px, py, vx, vy]
noise_std = np.array([0.0, 0.0, 0.1, 0.1])


# === FUNCTIONS ===
# Function to take a nonlinear time step:
def nonlinear_step(x, u, dt, c, add_noise=False, noise_std=None):
    """
    Propagate nonlinear dynamics with optional process noise.

    Args:
        x: Current state
        u: Control input
        dt: Time step
        c: Drag coefficient
        add_noise: Whether to add process noise
        noise_std: Standard deviation of noise for each state [std_px, std_py, std_vx, std_vy]
    """

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

    # Deterministic next state:
    x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # Add process noise if requested:
    if add_noise and noise_std is not None:
        # Generate Gaussian noise for each state
        noise = np.random.randn(4) * noise_std
        x_next += noise

    return x_next


def compute_u_ref(x_current, x_next, dt, c):
    """Compute reference control from trajectory."""
    # Extract states
    px, py, vx, vy = x_current
    px_next, py_next, vx_next, vy_next = x_next

    # Compute velocity derivatives from finite differences
    vx_dot = (vx_next - vx) / dt
    vy_dot = (vy_next - vy) / dt

    sqrt_term = np.sqrt(vx ** 2 + vy ** 2)
    ux_ref = vx_dot + c * vx * sqrt_term
    uy_ref = vy_dot + c * vy * sqrt_term

    return np.array([ux_ref, uy_ref])


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


# Discretize system to get:
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


# Linearized discrete step (for LTV - must match the discretization used in linearization):
def linearized_step(x, u, A_d, B_d, g_d):
    return A_d @ x + B_d @ u + g_d


def run_single_simulation(method_name, noise_std=None, c=2.0, dt=0.01, f=50, v=20,
                          W3=None, W4=None, x0=None, traj=None, u_min=None, u_max=None,
                          n_sim=None, lin_ctrl=False, true_init=False, C=None, r=None, m=None):
    """
    Run a single MPC simulation.

    Returns:
        x_sim: State trajectory (n_sim x 4)
        u_sim: Control trajectory (n_sim x 2)
        time_elapsed: Computation time
    """
    add_noise = noise_std is not None

    if method_name == 'LTI':
        # Linearize and discretize at trim point
        A_c, B_c, C_c, g_c = lin_cts(x0, np.zeros(2), c)
        A_lti, B_lti, g_lti = disc_sys(A_c, B_c, g_c, dt)

        # Initialize variables:
        x_sim = np.zeros((n_sim, 4))
        u_sim = np.zeros((n_sim, 2))

        # Apply initial condition:
        if true_init:
            x_current = traj[0, :]
        else:
            x_current = x0.copy()

        # Initialize MPC object:
        mpc = MPC(None, None, None, f, v, W3, W4, x_current,
                  traj, u_min, u_max, 'nonlinear', 'LTI')

        # Solve MPC problem:
        start = time.perf_counter()
        for k in range(n_sim):
            # Solve:
            mpc.control_inputs(A_lti, B_lti, C_c)

            # Extract control:
            u_k = mpc.inputs[-1].flatten()

            # Propagate true nonlinear dynamics:
            x_current = nonlinear_step(x_current, u_k, dt, c, add_noise, noise_std)

            # Extract states:
            mpc.states.append(x_current.reshape(-1, 1))

            # Assign current step:
            x_sim[k, :] = x_current
            u_sim[k, :] = u_k

        time_elapsed = time.perf_counter() - start

    elif method_name == 'LTV':
        # Initialize states:
        x_sim = np.zeros((n_sim, 4))
        u_sim = np.zeros((n_sim, 2))

        # Apply initial condition:
        if true_init:
            x_k = traj[0].copy()
        else:
            x_k = x0.copy()

        # Build MPC object:
        mpc = MPC(None, None, None, f, v, W3, W4, x_k, traj, u_min, u_max, 'nonlinear', 'LTV')

        # Start timer:
        start = time.perf_counter()

        # Use controller:
        for i in range(n_sim):
            # Extract reference state trajectory:
            x_ref_seq = [traj[i + j].copy() for j in range(f + 1)]

            # Initialize sequential matrices and nominal trajectories:
            A_seq = []
            B_seq = []
            C_seq = []
            r_seq = []
            u_nom_seq = []

            # Linearize about reference trajectory:
            for j in range(f):
                # Decide whether to create nominal control about trajectory or zeros:
                if lin_ctrl:
                    # Compute nominal control from reference trajectory
                    u_nom = compute_u_ref(x_ref_seq[j], x_ref_seq[j + 1], dt, c)
                else:
                    # Linearize around 0 nominal control:
                    u_nom = np.zeros((m,))

                # Store nominal control:
                u_nom_seq.append(u_nom)

                # Linearize continuous dynamics at (x_ref[j], u_nom):
                A_c, B_c, C_c, g_c = lin_cts(x_ref_seq[j], u_nom, c)

                # Discretize to get the discrete Jacobians:
                A_d, B_d, g_d = disc_sys(A_c, B_c, g_c, dt)

                # Compute r_k = f(x_ref[j], u_nom) - x_ref[j+1]
                f_at_nom = linearized_step(x_ref_seq[j], u_nom, A_d, B_d, g_d)
                r_j = f_at_nom - x_ref_seq[j + 1]

                # Store in sequences:
                A_seq.append(A_d)
                B_seq.append(B_d)
                C_seq.append(C_c)
                r_seq.append(r_j)

            # Solve MPC:
            if lin_ctrl:
                mpc.control_inputs(A_seq, B_seq, C_seq, x_ref_seq, r_seq, u_nom_seq)
                delta_u_k = mpc.inputs[-1].flatten()
                u_k = u_nom_seq[0] + delta_u_k
            else:
                mpc.control_inputs(A_seq, B_seq, C_seq, x_ref_seq, r_seq)
                u_k = mpc.inputs[-1].flatten()

            # Propagate TRUE nonlinear dynamics:
            x_k = nonlinear_step(x_k, u_k, dt, c, add_noise, noise_std)

            # Feed true state back to MPC:
            mpc.states.append(x_k.reshape(-1, 1))

            # Store current time step:
            x_sim[i, :] = x_k
            u_sim[i, :] = u_k

        time_elapsed = time.perf_counter() - start

    elif method_name == 'NL':
        # Make variables:
        W3_ca = ca.DM(W3)
        W4_ca = ca.DM(W4)
        C_ca = ca.DM(C)

        # Build MPC problem:
        solver_nl, lbx, ubx, lbg, ubg = build_nlmpc(c, dt, f, v, C_ca, nx=4, nu=2, ny=4,
                                                    umin=u_min, umax=u_max, W3=W3_ca, W4=W4_ca)

        # Initialize variables:
        x_sim = np.zeros((n_sim, 4))
        u_sim = np.zeros((n_sim, 2))

        # Apply initial condition:
        if true_init:
            x_current = traj[0, :]
        else:
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
            x_current = nonlinear_step(x_current, u_opt, dt, c, add_noise, noise_std)

            # Assign current values:
            x_sim[k, :] = x_current
            u_sim[k, :] = u_opt

        time_elapsed = time.perf_counter() - start

    else:
        raise ValueError(f"Unknown method: {method_name}")

    return x_sim, u_sim, time_elapsed


def run_monte_carlo_simulation(method_name, n_trials=50, noise_std=None, **sim_params):
    """
    Run multiple trials with process noise and collect statistics.

    Args:
        method_name: 'LTI', 'LTV', or 'NL'
        n_trials: Number of Monte Carlo runs
        noise_std: Noise standard deviation [std_px, std_py, std_vx, std_vy]
        **sim_params: Simulation parameters (c, dt, f, v, W3, W4, etc.)

    Returns:
        Dictionary with mean and std of tracking error and control cost
    """
    errors = []
    costs = []
    times = []
    all_trajectories = []

    for trial in range(n_trials):
        # Set random seed for reproducibility
        np.random.seed(trial)

        # Run single simulation
        x_sim, u_sim, time_elapsed = run_single_simulation(
            method_name, noise_std=noise_std, **sim_params
        )

        # Compute metrics
        error = np.sum((x_sim[:, :2] - sim_params['traj'][:sim_params['n_sim'], :2]) ** 2)
        cost = np.sum(u_sim ** 2)

        errors.append(error)
        costs.append(cost)
        times.append(time_elapsed)
        all_trajectories.append({'x': x_sim, 'u': u_sim})

        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial + 1}/{n_trials} completed")

    return {
        'error_mean': np.mean(errors),
        'error_std': np.std(errors),
        'cost_mean': np.mean(costs),
        'cost_std': np.std(costs),
        'time_mean': np.mean(times),
        'time_std': np.std(times),
        'all_errors': errors,
        'all_costs': costs,
        'all_times': times,
        'trajectories': all_trajectories
    }


def plot_monte_carlo_results(results, traj, n_sim, dt, methods_to_run):
    """Plot mean trajectories with confidence intervals."""
    time_vec = np.arange(n_sim) * dt

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(
        f'Monte Carlo MPC Comparison (Mean ± 1σ, {len(results[methods_to_run[0].lower()]["trajectories"])} trials)',
        fontsize=16, fontweight='bold')

    colors = {'lti': 'C0', 'ltv': 'C1', 'nonlinear': 'C2'}

    for method in methods_to_run:
        m = method.lower()
        trajectories = results[m]['trajectories']
        color = colors.get(m, 'C3')

        # Stack all trajectories
        all_px = np.array([t['x'][:, 0] for t in trajectories])
        all_py = np.array([t['x'][:, 1] for t in trajectories])
        all_vx = np.array([t['x'][:, 2] for t in trajectories])
        all_vy = np.array([t['x'][:, 3] for t in trajectories])
        all_ux = np.array([t['u'][:, 0] for t in trajectories])
        all_uy = np.array([t['u'][:, 1] for t in trajectories])

        # Compute mean and std
        px_mean, px_std = np.mean(all_px, axis=0), np.std(all_px, axis=0)
        py_mean, py_std = np.mean(all_py, axis=0), np.std(all_py, axis=0)
        vx_mean, vx_std = np.mean(all_vx, axis=0), np.std(all_vx, axis=0)
        vy_mean, vy_std = np.mean(all_vy, axis=0), np.std(all_vy, axis=0)
        ux_mean, ux_std = np.mean(all_ux, axis=0), np.std(all_ux, axis=0)
        uy_mean, uy_std = np.mean(all_uy, axis=0), np.std(all_uy, axis=0)

        # Plot px
        axes[0, 0].plot(time_vec, px_mean, label=f'{m.upper()}', linewidth=2, color=color)
        axes[0, 0].fill_between(time_vec, px_mean - px_std, px_mean + px_std, alpha=0.2, color=color)

        # Plot py
        axes[0, 1].plot(time_vec, py_mean, label=f'{m.upper()}', linewidth=2, color=color)
        axes[0, 1].fill_between(time_vec, py_mean - py_std, py_mean + py_std, alpha=0.2, color=color)

        # Plot vx
        axes[1, 0].plot(time_vec, vx_mean, label=f'{m.upper()}', linewidth=2, color=color)
        axes[1, 0].fill_between(time_vec, vx_mean - vx_std, vx_mean + vx_std, alpha=0.2, color=color)

        # Plot vy
        axes[1, 1].plot(time_vec, vy_mean, label=f'{m.upper()}', linewidth=2, color=color)
        axes[1, 1].fill_between(time_vec, vy_mean - vy_std, vy_mean + vy_std, alpha=0.2, color=color)

        # Plot ux
        axes[2, 0].plot(time_vec, ux_mean, label=f'{m.upper()}', linewidth=2, color=color)
        axes[2, 0].fill_between(time_vec, ux_mean - ux_std, ux_mean + ux_std, alpha=0.2, color=color)

        # Plot uy
        axes[2, 1].plot(time_vec, uy_mean, label=f'{m.upper()}', linewidth=2, color=color)
        axes[2, 1].fill_between(time_vec, uy_mean - uy_std, uy_mean + uy_std, alpha=0.2, color=color)

    # Reference trajectories
    axes[0, 0].plot(time_vec, traj[:n_sim, 0], 'k--', label='Reference', linewidth=2)
    axes[0, 1].plot(time_vec, traj[:n_sim, 1], 'k--', label='Reference', linewidth=2)
    axes[1, 0].plot(time_vec, traj[:n_sim, 2], 'k--', label='Reference', linewidth=2)
    axes[1, 1].plot(time_vec, traj[:n_sim, 3], 'k--', label='Reference', linewidth=2)

    # Labels
    axes[0, 0].set_ylabel('px (m)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_ylabel('py (m)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_ylabel('vx (m/s)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_ylabel('vy (m/s)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 0].set_ylabel('ux (N)')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].set_ylabel('uy (N)')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Second figure: Box plots and tracking error
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
    fig2.suptitle('Performance Statistics', fontsize=16, fontweight='bold')

    # Box plots for errors and costs
    error_data = [results[m.lower()]['all_errors'] for m in methods_to_run]
    cost_data = [results[m.lower()]['all_costs'] for m in methods_to_run]

    bp1 = axes2[0, 0].boxplot(error_data, labels=[m.upper() for m in methods_to_run], patch_artist=True)
    axes2[0, 0].set_ylabel('Tracking Error')
    axes2[0, 0].set_yscale('log')
    axes2[0, 0].grid(True, alpha=0.3)
    axes2[0, 0].set_title('Trajectory Error Distribution')

    # Color the boxes
    for patch, method in zip(bp1['boxes'], methods_to_run):
        patch.set_facecolor(colors.get(method.lower(), 'C3'))
        patch.set_alpha(0.6)

    bp2 = axes2[0, 1].boxplot(cost_data, labels=[m.upper() for m in methods_to_run], patch_artist=True)
    axes2[0, 1].set_ylabel('Control Cost')
    axes2[0, 1].set_yscale('log')
    axes2[0, 1].grid(True, alpha=0.3)
    axes2[0, 1].set_title('Control Cost Distribution')

    for patch, method in zip(bp2['boxes'], methods_to_run):
        patch.set_facecolor(colors.get(method.lower(), 'C3'))
        patch.set_alpha(0.6)

    # Tracking error over time (mean)
    for method in methods_to_run:
        m = method.lower()
        trajectories = results[m]['trajectories']

        # Compute tracking errors
        all_errors = []
        for traj_data in trajectories:
            err = np.linalg.norm(traj_data['x'][:, :2] - traj[:n_sim, :2], axis=1)
            all_errors.append(err)

        all_errors = np.array(all_errors)
        err_mean = np.mean(all_errors, axis=0)
        err_std = np.std(all_errors, axis=0)

        color = colors.get(m, 'C3')
        axes2[1, 0].plot(time_vec, err_mean, label=f'{m.upper()}', linewidth=2, color=color)
        axes2[1, 0].fill_between(time_vec, err_mean - err_std, err_mean + err_std, alpha=0.2, color=color)

    axes2[1, 0].set_ylabel('Tracking Error (m)')
    axes2[1, 0].set_xlabel('Time (s)')
    axes2[1, 0].set_yscale('log')
    axes2[1, 0].grid(True, alpha=0.3)
    axes2[1, 0].legend()
    axes2[1, 0].set_title('Tracking Error Over Time')

    # Cumulative control cost over time (mean)
    for method in methods_to_run:
        m = method.lower()
        trajectories = results[m]['trajectories']

        all_cum_costs = []
        for traj_data in trajectories:
            cum_cost = np.cumsum(np.sum(traj_data['u'] ** 2, axis=1))
            all_cum_costs.append(cum_cost)

        all_cum_costs = np.array(all_cum_costs)
        cost_mean = np.mean(all_cum_costs, axis=0)
        cost_std = np.std(all_cum_costs, axis=0)

        color = colors.get(m, 'C3')
        axes2[1, 1].plot(time_vec, cost_mean, label=f'{m.upper()}', linewidth=2, color=color)
        axes2[1, 1].fill_between(time_vec, cost_mean - cost_std, cost_mean + cost_std, alpha=0.2, color=color)

    axes2[1, 1].set_ylabel('Cumulative Control Cost')
    axes2[1, 1].set_xlabel('Time (s)')
    axes2[1, 1].set_yscale('log')
    axes2[1, 1].grid(True, alpha=0.3)
    axes2[1, 1].legend()
    axes2[1, 1].set_title('Cumulative Control Cost Over Time')

    plt.tight_layout()
    plt.show()


# === SETUP SIMULATION ===
# ------ IMPORTANT TUNING PARAMETERS ------
# Horizon variables:
f = 50
v = 20

# Weights:
Q0 = 0.1 * np.eye(2)
Q = np.eye(2)
P_full = 1000 * np.diag([1000, 1000, 1000, 1000])

# Options:
lin_ctrl = False
true_init = False
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

# Package simulation parameters:
sim_params = {
    'c': c, 'dt': dt, 'f': f, 'v': v,
    'W3': W3, 'W4': W4, 'x0': x0, 'traj': traj,
    'u_min': u_min, 'u_max': u_max, 'n_sim': n_sim,
    'lin_ctrl': lin_ctrl, 'true_init': true_init,
    'C': C, 'r': r, 'm': m
}

results = {}

# === RUN SIMULATIONS ===
if run_monte_carlo:
    print("\n" + "=" * 80)
    print(f"Running Monte Carlo Simulations ({n_trials} trials per method)")
    print(f"Process Noise: px={noise_std[0]:.3f}, py={noise_std[1]:.3f}, vx={noise_std[2]:.3f}, vy={noise_std[3]:.3f}")
    print("=" * 80)

    for method in methods_to_run:
        print(f"\n{method.upper()} MPC:")
        mc_results = run_monte_carlo_simulation(method, n_trials, noise_std, **sim_params)
        results[method.lower()] = mc_results

    # === PRINT SUMMARY ===
    print("\n" + "=" * 90)
    print("MONTE CARLO SIMULATION RESULTS (Mean ± Std)")
    print("=" * 90)
    print(f"{'Method':<12} {'Runtime (s)':<25} {'Traj Error':<25} {'Control Cost':<25}")
    print("-" * 90)
    for method in methods_to_run:
        m = method.lower()
        print(f"{m.upper():<12} "
              f"{results[m]['time_mean']:<12.3f} ± {results[m]['time_std']:<10.3f} "
              f"{results[m]['error_mean']:<12.2f} ± {results[m]['error_std']:<10.2f} "
              f"{results[m]['cost_mean']:<12.2f} ± {results[m]['cost_std']:<10.2f}")
    print("=" * 90)

    # Plot results
    print("\nGenerating plots...")
    plot_monte_carlo_results(results, traj, n_sim, dt, methods_to_run)

else:
    # Run single deterministic simulation
    print("\n" + "=" * 80)
    print("Running Single Deterministic Simulations")
    print("=" * 80)

    for method in methods_to_run:
        print(f"\n{method.upper()} MPC:")
        x_sim, u_sim, time_elapsed = run_single_simulation(method, noise_std=None, **sim_params)

        error = np.sum((x_sim[:, :2] - traj[:n_sim, :2]) ** 2)
        cost = np.sum(u_sim ** 2)

        results[method.lower()] = {
            'x': x_sim,
            'u': u_sim,
            'time': time_elapsed,
            'error': error,
            'cost': cost
        }

    # Print summary
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"{'Method':<15} {'Runtime (s)':<15} {'Traj Error':<15} {'Control Cost':<15}")
    print("-" * 60)
    for method in methods_to_run:
        m = method.lower()
        print(f"{m.upper():<15} {results[m]['time']:<15.3f} {results[m]['error']:<15.2f} {results[m]['cost']:<15.2f}")
    print("=" * 60)

    # Simple plots for deterministic case
    time_vec = np.arange(n_sim) * dt
    fig, axes = plt.subplots(4, 2, figsize=(14, 10))
    fig.suptitle('MPC Comparison (Deterministic)', fontsize=16, fontweight='bold')

    for i, (state_name, ylabel) in enumerate([('px', 'px (m)'), ('py', 'py (m)'),
                                              ('vx', 'vx (m/s)'), ('vy', 'vy (m/s)')]):
        axes[i, 0].plot(time_vec, traj[:n_sim, i], 'k--', linewidth=2, label='Reference')
        for method in methods_to_run:
            m = method.lower()
            axes[i, 0].plot(time_vec, results[m]['x'][:, i], label=m.upper(), alpha=0.8)
        axes[i, 0].set_ylabel(ylabel)
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].legend()

    axes[3, 0].set_xlabel('Time (s)')

    # Controls
    for i, ylabel in enumerate(['ux (N)', 'uy (N)']):
        for method in methods_to_run:
            m = method.lower()
            axes[i, 1].plot(time_vec, results[m]['u'][:, i], label=m.upper(), alpha=0.8)
        axes[i, 1].axhline(u_max[i], color='k', linestyle=':', alpha=0.5)
        axes[i, 1].axhline(u_min[i], color='k', linestyle=':', alpha=0.5)
        axes[i, 1].set_ylabel(ylabel)
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].legend()

    # Tracking error
    for method in methods_to_run:
        m = method.lower()
        err = np.linalg.norm(results[m]['x'][:, :2] - traj[:n_sim, :2], axis=1)
        axes[2, 1].plot(time_vec, err, label=m.upper(), alpha=0.8)
    axes[2, 1].set_ylabel('Tracking Error (m)')
    axes[2, 1].set_yscale('log')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()

    # Control cost
    for method in methods_to_run:
        m = method.lower()
        cost_inst = np.cumsum(np.sum(results[m]['u'] ** 2, axis=1))
        axes[3, 1].plot(time_vec, cost_inst, label=m.upper(), alpha=0.8)
    axes[3, 1].set_xlabel('Time (s)')
    axes[3, 1].set_ylabel('Control Cost (Cum.)')
    axes[3, 1].set_yscale('log')
    axes[3, 1].grid(True, alpha=0.3)
    axes[3, 1].legend()

    plt.tight_layout()
    plt.show()