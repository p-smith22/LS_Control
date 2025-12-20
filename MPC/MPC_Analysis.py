# Import packages:
import numpy as np
import matplotlib.pyplot as plt
import os
from src.utils import plot_heatmap

# Load sweep files:
save_dir = "results"
files = [f for f in os.listdir(save_dir) if f.endswith(".npy")]
if len(files) == 0:
    raise RuntimeError("No result files found in results/")

# Initialize lists:
f_list, v_list = [], []
runtime_list, traj_error_list, ctrl_cost_list = [], [], []

# Load data into arrays:
for fname in files:
    data = np.load(os.path.join(save_dir, fname), allow_pickle=True).item()
    f_list.append(data["f"])
    v_list.append(data["v"])
    runtime_list.append(data["runtime"])
    traj_error_list.append(data["traj_error"])
    ctrl_cost_list.append(data["ctrl_cost"])

# Store unique prediction/control horizon values:
f_unique = np.unique(f_list)
v_unique = np.unique(v_list)

# --- HEATMAPS ---
# Build grids for heatmap:
runtime_grid = np.full((len(f_unique), len(v_unique)), np.nan)
traj_error_grid = np.full_like(runtime_grid, np.nan)
ctrl_cost_grid = np.full_like(runtime_grid, np.nan)
for k in range(len(files)):
    i = np.where(f_unique == f_list[k])[0][0]
    j = np.where(v_unique == v_list[k])[0][0]
    runtime_grid[i, j] = runtime_list[k]
    traj_error_grid[i, j] = traj_error_list[k]
    ctrl_cost_grid[i, j] = ctrl_cost_list[k]

# Plot heatmaps:
plot_heatmap(runtime_grid, v_unique, f_unique,"Computational Time [s]", cmap="plasma")
plot_heatmap(traj_error_grid, v_unique, f_unique, "Trajectory Error (last 5s)", cmap="plasma")
plot_heatmap(ctrl_cost_grid, v_unique, f_unique,   "Control Effort", cmap="plasma")

# --- CONTROLS AND TRAJECTORY ---
# Select trajectories:
selected_runs = [
    (10, 10),
    (20, 20),
]

# Define colors for trajectories:
colors = plt.cm.cool(np.linspace(0, 1, len(selected_runs)))

# Fetch run data:
runs = []
for (f, v) in selected_runs:
    fname = os.path.join(save_dir, f"mpc_f{f}_v{v}.npy")
    if not os.path.exists(fname):
        print(f"WARNING: {fname} not found, skipping.")
        continue
    data = np.load(fname, allow_pickle=True).item()
    runs.append((f, v, data))
if len(runs) == 0:
    raise RuntimeError("No selected runs found.")

# Get time vector from first run:
dt = 0.01
time = runs[0][2]["trajectory"].shape[0] * dt
t = np.linspace(0, time, runs[0][2]["trajectory"].shape[0])

# Initialize trajectory plots:
fig_states, axes_states = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes_states = axes_states.flatten()

# Define labels:
state_labels = ["$p_x$ (m)", "$p_y$ (m)", "$v_x$ (m/s)", "$v_y$ (m/s)"]

# Generate reference trajectory for plotting:
t_ref = np.linspace(0, time, len(t))
ref_trajectory = np.zeros((len(t), 4))
for i in range(len(t)):
    ref_trajectory[i, 0] = 4 * t_ref[i]  # px = 4*t
    ref_trajectory[i, 1] = 30            # py = 30
    ref_trajectory[i, 2] = 4             # vx = 4
    ref_trajectory[i, 3] = 0             # vy = 0

# Plot:
for (f, v, data), color in zip(runs, colors):
    x = data["trajectory"]
    t_data = np.linspace(0, x.shape[0] * dt, x.shape[0])
    for i in range(4):
        axes_states[i].plot(t_data, x[:, i], color=color, linewidth=2, label=f"f={f}, v={v}")

# Reference lines:
for i in range(4):
    axes_states[i].plot(t_ref, ref_trajectory[:, i], color="black", linestyle="--", linewidth=1.5, label="Reference" if i == 0 else None)

# Set labels:
for ax, label in zip(axes_states, state_labels):
    ax.set_ylabel(label)
    ax.grid(True)
axes_states[2].set_xlabel("Time (s)")
axes_states[3].set_xlabel("Time (s)")

# Plotting parameters:
axes_states[0].legend(loc="best", fontsize=9, frameon=True)
fig_states.suptitle("State Trajectories Overlay", fontsize=14)
fig_states.tight_layout(rect=[0, 0, 1, 0.95])

# Initialize control plots:
fig_ctrl, axes_ctrl = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Plot:
for (f, v, data), color in zip(runs, colors):
    u = data["control"]
    t_ctrl = np.linspace(0, u.shape[0] * dt, u.shape[0])
    axes_ctrl[0].plot(t_ctrl, u[:, 0], color=color, linewidth=2, label=f"f={f}, v={v}")
    axes_ctrl[1].plot(t_ctrl, u[:, 1], color=color, linewidth=2)

# Control limits:
axes_ctrl[0].axhline(20, color="black", linestyle="--", linewidth=1.5, label="Limit")
axes_ctrl[0].axhline(-20, color="black", linestyle="--", linewidth=1.5)
axes_ctrl[1].axhline(20, color="black", linestyle="--", linewidth=1.5)
axes_ctrl[1].axhline(-20, color="black", linestyle="--", linewidth=1.5)

# Set labels:
axes_ctrl[0].set_ylabel("$u_x$ (m/s²)")
axes_ctrl[1].set_ylabel("$u_y$ (m/s²)")
axes_ctrl[1].set_xlabel("Time (s)")
for ax in axes_ctrl:
    ax.grid(True)

# Plotting parameters:
axes_ctrl[0].legend(loc="best", fontsize=9, frameon=True)
fig_ctrl.suptitle("Control Inputs Overlay", fontsize=14)
fig_ctrl.tight_layout(rect=[0, 0, 1, 0.95])

# --- 2D X-Y TRAJECTORY PLOT ---
fig_xy, ax_xy = plt.subplots(figsize=(10, 8))

# Plot trajectories:
for (f, v, data), color in zip(runs, colors):
    x = data["trajectory"]
    ax_xy.plot(x[:, 0], x[:, 1], color=color, linewidth=2, label=f"f={f}, v={v}")

# Plot reference line (constant y = 30):
x_max = max([data["trajectory"][-1, 0] for _, _, data in runs])
ax_xy.plot([0, x_max], [30, 30], 'k--', linewidth=1.5, label='Reference')

# Set labels and formatting:
ax_xy.set_xlabel("$p_x$ (m)", fontsize=12)
ax_xy.set_ylabel("$p_y$ (m)", fontsize=12)
ax_xy.grid(True, alpha=0.3)
ax_xy.legend(loc="best", fontsize=10, frameon=True)
ax_xy.set_title("2D Trajectory Overlay", fontsize=14)
fig_xy.tight_layout()

# Show graphs:
plt.show()
