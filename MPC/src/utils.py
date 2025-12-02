# Import packages:
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from rbf.interpolate import RBFInterpolant

# Discretization functino:
def discretize_system(a_cts, b_cts, dt):

    # Fetch dimensions:
    n = a_cts.shape[0]
    m = b_cts.shape[1]

    # Build augmented matrix:
    M = np.zeros((n + m, n + m))
    M[:n, :n] = a_cts * dt
    M[:n, n:] = b_cts * dt

    # Discretize:
    exp_m = expm(M)
    a_disc = exp_m[:n, :n]
    b_disc = exp_m[:n, n:]

    # Return discretized matrices:
    return a_disc, b_disc

# Simulate the discrete linear system:
def simulate(a, b, c, u, x0, t_end):

    # Ensure shape:
    u = u.reshape(-1, 1)
    x0 = x0.flatten()

    # Initialize matrices:
    n_tsteps = u.shape[1]
    n = a.shape[0]
    r = c.shape[0]
    x = np.zeros(shape=(n, n_tsteps + 1))
    y = np.zeros(shape=(r, n_tsteps))
    t = np.linspace(0, t_end, n_tsteps)

    # Initial condition:
    x[:, 0] = x0

    # Simulation loop:
    for k in range(n_tsteps):
        x[:, k + 1] = a @ x[:, k] + b @ u[k]
        y[:, k] = c @ x[:, k]

    # Return states:
    return y, x

def plot_graphs(time_ctrl, y, y_des, ctrl, opt, bounds):

    # Plot spring dashpot system:
    if opt == 'spring_dashpot':

        # Plot desired vs actual trajectory:
        lab_opts = ['y1', 'y2', 'y3', 'y4']
        vars = len(y_des[0, :])
        if vars < 2:
            plt.figure()
            plt.plot(time_ctrl, y[:, 0], label='Controlled')
            plt.plot(time_ctrl, y_des[:, 0], label='Desired')
            plt.ylabel(lab_opts[0])
            plt.legend(title="Trajectory")
            plt.xlabel('Time (s)')
        else:
            fig, axs = plt.subplots(vars, 1, sharex=True)
            for i in range(vars):
                axs[i].plot(time_ctrl, y[:, i], label='Controlled')
                axs[i].plot(time_ctrl, y_des[:, i], label='Desired')
                axs[i].set_ylabel(lab_opts[i])
            axs[-1].legend(title="Trajectory")
            axs[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.xlim([0, time_ctrl[-1]])
        plt.savefig('./output/Trajectory_SD.png', dpi=600)

        # Plot control values:
        plt.figure()
        plt.plot(time_ctrl, ctrl, linewidth=4)
        if bounds[0] is not None:
            plt.axhline(y=bounds[0], color='r', linestyle='--')
        if bounds[1] is not None:
            plt.axhline(y=bounds[1], color='r', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Control Input')
        plt.legend()
        plt.xlim([0, time_ctrl[-1]])
        plt.savefig('./output/Controls_SD.png', dpi=600)

    # Plot AVL system:
    if opt == 'AVL':

        # Plot desired vs actual trajectory:
        lab_opts = ['u (m/s)', 'w (m/s)', 'q (deg/s)', 'theta (deg)', 'v (m/s)', 'p (deg/s)', 'r (deg/s)',
                    'phi (deg)', 'x (m)', 'y (m)', 'z (m)', 'psi (deg)']

        # Plot position:
        fig, ax1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle('Position States', fontsize=14)
        position_labels = ['x (m)', 'y (m)', 'z (m)']
        position_idx = [8, 9, 10]
        for i, idx in enumerate(position_idx):
            ax1[i].plot(time_ctrl, y[:, idx], linewidth=4, label='Controlled')
            ax1[i].plot(time_ctrl, y_des[:, idx], linewidth=4, label='Desired')
            ax1[i].set_ylabel(position_labels[i], fontsize=12)
            ax1[i].grid(True, alpha=0.3)
            ax1[i].legend()
        ax1[-1].set_xlabel('Time (s)', fontsize=12)
        plt.tight_layout()
        plt.xlim([0, time_ctrl[-1]])
        plt.savefig('./output/Positions_AVL.png', dpi=600)

        # Plot velocities:
        fig, ax2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle('Velocity States', fontsize=14)
        velocity_labels = ['u (m/s)', 'v (m/s)', 'w (m/s)']
        velocity_idx = [0, 4, 1]
        for i, idx in enumerate(velocity_idx):
            ax2[i].plot(time_ctrl, y[:, idx], linewidth=4, label='Controlled')
            ax2[i].plot(time_ctrl, y_des[:, idx], linewidth=4, label='Desired')
            ax2[i].set_ylabel(velocity_labels[i], fontsize=12)
            ax2[i].grid(True, alpha=0.3)
            ax2[i].legend()
        ax2[-1].set_xlabel('Time (s)', fontsize=12)
        plt.tight_layout()
        plt.xlim([0, time_ctrl[-1]])
        plt.savefig('./output/Velocities_AVL.png', dpi=600)

        # Plot Euler angles:
        fig, ax3 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle('Euler Angles', fontsize=14)
        angle_labels = ['\u0398 (deg)', '\u03A6 (deg)', '\u03A8 (deg)']
        angle_idx = [3, 7, 11]
        for i, idx in enumerate(angle_idx):
            ax3[i].plot(time_ctrl, np.rad2deg(y[:, idx]), linewidth=4, label='Controlled')
            ax3[i].plot(time_ctrl, np.rad2deg(y_des[:, idx]), linewidth=4, label='Desired')
            ax3[i].set_ylabel(angle_labels[i], fontsize=12)
            ax3[i].grid(True, alpha=0.3)
            ax3[i].legend()
        ax3[-1].set_xlabel('Time (s)', fontsize=12)
        plt.tight_layout()
        plt.xlim([0, time_ctrl[-1]])
        plt.savefig('./output/EulerAngles_AVL.png', dpi=600)

        # Plot Euler rates:
        fig, ax4 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle('Angular Rates', fontsize=14)
        rate_labels = ['q (deg/s)', 'p (deg/s)', 'r (deg/s)']
        rate_idx = [2, 5, 6]
        for i, idx in enumerate(rate_idx):
            ax4[i].plot(time_ctrl, np.rad2deg(y[:, idx]), linewidth=4, label='Controlled')
            ax4[i].plot(time_ctrl, np.rad2deg(y_des[:, idx]), linewidth=4, label='Desired')
            ax4[i].set_ylabel(rate_labels[i], fontsize=12)
            ax4[i].grid(True, alpha=0.3)
            ax4[i].legend()
        ax4[-1].set_xlabel('Time (s)', fontsize=12)
        plt.tight_layout()
        plt.xlim([0, time_ctrl[-1]])
        plt.savefig('./output/EulerRates_AVL.png', dpi=600)

        fig, ax5 = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        fig.suptitle('Control Inputs', fontsize=14)
        control_labels = ['Camber (deg)', 'Aileron (deg)', 'Elevator (deg)', 'Rudder (deg)']
        for i in range(ctrl.shape[1]):
            ax5[i].plot(time_ctrl, np.rad2deg(ctrl[:, i]), linewidth=4)
            if bounds[0, i] is not None:
                ax5[i].axhline(np.rad2deg(bounds[0, i]), color='r', linestyle='--')
            if bounds[1, i] is not None:
                ax5[i].axhline(np.rad2deg(bounds[1, i]), color='r', linestyle='--')
            ax5[i].set_ylabel(control_labels[i], fontsize=12)
            ax5[i].grid(True, alpha=0.3)
        ax5[-1].set_xlabel('Time (s)', fontsize=12)
        plt.tight_layout()
        plt.xlim([0, time_ctrl[-1]])
        plt.savefig('./output/Controls_AVL.png', dpi=600)

def train_rbf():

    # Load data:
    a = np.load('./data/A_trainer_5_large.npy')
    b = np.load('./data/B_trainer_5_large.npy')
    trainer_vals = np.load('./data/sobolsequence_5_large.npy')

    # Formatting:
    iterations = len(trainer_vals[:, 0])
    a_reshape = a.reshape(iterations, -1)
    b_reshape = b.reshape(iterations, -1)

    # Set interpolated matrices:
    interp_a = RBFInterpolant(trainer_vals, a_reshape)
    interp_b = RBFInterpolant(trainer_vals, b_reshape)
    return interp_a, interp_b
