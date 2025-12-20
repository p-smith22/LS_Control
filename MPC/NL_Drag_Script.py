# Import packages:
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
from src.MPC import MPC
from src.utils import discretize_system, plot_graphs

# Linearize:
def linearization(vx, vy, c):
    sqrt_term = np.sqrt(vx ** 2 + vy ** 2)
    if sqrt_term == 0.0:
        sqrt_term = 1e-2

    dvx_dvx = -c * (sqrt_term + vx ** 2 / sqrt_term)
    dvx_dvy = -c * vx * vy / sqrt_term
    dvy_dvx = -c * vy * vx / sqrt_term
    dvy_dvy = -c * (sqrt_term + vy ** 2 / sqrt_term)

    a_mat = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, dvx_dvx, dvx_dvy],
        [0, 0, dvy_dvx, dvy_dvy]
    ])
    b_mat = np.array([
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1]
    ])
    c_mat = np.eye(4)

    return a_mat, b_mat, c_mat

# Nonlinear timestep:
def nonlinear_step(x, u, dt, c):
    px, py, vx, vy = x
    ux, uy = u
    v = np.sqrt(vx ** 2 + vy ** 2)
    dx = np.array([
        vx,
        vy,
        ux - c * vx * v,
        uy - c * vy * v
    ])
    return x + dt * dx

# Simulation parameters:
dt = 0.01  # s
t_end = 60  # s
n_tsteps = int(t_end / dt)
time_vec = np.linspace(0, t_end, n_tsteps)

# MPC Sweep parameters:
f_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
v_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Problem parameters:
c = 0.4
x0 = np.array([0, 0, 4, 0])  # Start at origin with zero velocity
u_max = np.array([20, 20])
u_min = np.array([-20, -20])
cts_lin = True

# Weight scalers:
Q0_scaler = 1e2
Q_scaler = 1e3

# Weight matrices:
Q0 = 0.00001 * np.eye(2) * Q0_scaler
Q = 0.0001 * np.eye(2) * Q_scaler

P = np.zeros((4, 4))
P[0, 0] = 100  # px
P[1, 1] = 1000  # py
P[2, 2] = 100  # vx
P[3, 3] = 1000  # vy

# Desired trajectory - Step input to (20, 20) with zero velocity:
traj = np.zeros((n_tsteps, 4))
for i in range(n_tsteps):
    t = dt * i
    traj[i, 0] = 4 * t
    traj[i, 1] = 30
    traj[i, 2] = 4
    traj[i, 3] = 0

# Dimensions:
r = 4
m = Q0.shape[0]
n = P.shape[0]

# Create folder to store results:
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)
results = []

# Calculate total number for progress bar:
horizon_pairs = [(f, v) for f in f_list for v in v_list if v <= f]
total_runs = len(horizon_pairs)

# Sweep:
with tqdm(total=total_runs, desc="Sweep Progress") as pbar:
    for f in f_list:
        for v in v_list:
            if v > f:
                continue

            # Construct W1, W2, W3, W4:
            W1 = np.zeros((v * m, v * m))
            for i in range(v):
                W1[i * m:(i + 1) * m, i * m:(i + 1) * m] = np.eye(m)
                if i > 0:
                    W1[i * m:(i + 1) * m, (i - 1) * m:i * m] = -np.eye(m)

            W2 = np.zeros((v * m, v * m))
            for i in range(v):
                if i == 0:
                    Q_use = Q0
                else:
                    Q_use = Q
                W2[i * m:(i + 1) * m, i * m:(i + 1) * m] = Q_use

            W3 = W1.T @ W2 @ W1

            W4 = np.zeros((f * r, f * r))
            for i in range(f):
                W4[i * r:(i + 1) * r, i * r:(i + 1) * r] = P

            # Build MPC object:
            mpc = MPC(None, None, None, f, v, W3, W4, x0, traj, u_min, u_max, 'nonlinear')

            # Initialize:
            x_hist = np.zeros((n_tsteps - f, 4))
            u_hist = []
            x_k = x0.copy()
            vx, vy = x_k[2], x_k[3]

            # Start timer:
            start_time = time.perf_counter()

            # Simulation:
            for i in range(n_tsteps - f):
                if cts_lin:
                    A_cts, B_cts, C_cts = linearization(vx, vy, c)

                A, B = discretize_system(A_cts, B_cts, dt)

                mpc.control_inputs(A, B, C_cts)
                u_k = np.asarray(mpc.inputs[-1]).reshape(-1)

                x_k = nonlinear_step(x_k, u_k, dt, c)
                x_hist[i, :] = x_k
                u_hist.append(u_k)

                vx, vy = x_k[2], x_k[3]
                mpc.states.append(x_k.reshape(-1, 1))

            u_hist = np.array(u_hist)

            # Calculate runtime:
            runtime = time.perf_counter() - start_time

            # Compute errors and control cost (use last 5 seconds):
            window = int(5 / dt)
            x_tail = x_hist[-window:, :2]
            x_ref_tail = traj[-window:, :2]
            traj_error = np.sum((x_tail - x_ref_tail) ** 2)
            ctrl_cost = np.sum(u_hist ** 2)

            # Save results:
            result = {
                "f": f,
                "v": v,
                "runtime": runtime,
                "trajectory": x_hist,
                "control": u_hist,
                "traj_error": traj_error,
                "ctrl_cost": ctrl_cost
            }

            # Store:
            filename = f"{save_dir}/mpc_f{f}_v{v}.npy"
            np.save(filename, result)
            results.append(result)

            # Update progress bar:
            pbar.set_postfix({"f": f, "v": v, "runtime[s]": f"{runtime:.2f}"})
            pbar.update(1)

print("\nSweep complete! Results saved to:", save_dir)
