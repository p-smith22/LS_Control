"""
Controllability Gramian AVL Example — Multi-Ellipsoid Plot
"""

# Import packages:
import numpy as np
from rbf.interpolate import RBFInterpolant
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches


def make_projected_ellipsoid(U, S, idx):
    """
    Returns Xx, Xy, Xz meshes for plotting a projected ellipsoid.
    """

    n = U.shape[0]

    # Parametric sphere
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    sphere_points = np.vstack([
        xs.flatten(),
        ys.flatten(),
        zs.flatten()
    ])

    # Embed sphere into n-D
    q = np.zeros((n, sphere_points.shape[1]))
    q[0:3, :] = sphere_points

    # Transform to ellipsoid
    X_hd = U @ (np.diag(S) @ q)

    # Project to (idx) subspace
    X = X_hd[idx, :]

    # Reshape for surface plot
    Xx = X[0].reshape(xs.shape)
    Xy = X[1].reshape(ys.shape)
    Xz = X[2].reshape(zs.shape)

    return Xx, Xy, Xz

def plot_multiple_ellipsoids(ellipsoids, idx=[3, 7, 11]):
    """
    ellipsoids = list of tuples (U, S, label, color)
    idx        = state indices to project onto (length 3)
    """

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')

    legend_patches = []

    for (U, S, label, color) in ellipsoids:

        Xx, Xy, Xz = make_projected_ellipsoid(U, S, idx)

        ax.plot_surface(
            Xx, Xy, Xz,
            rstride=1, cstride=1,
            alpha=0.35,
            color=color,
            edgecolor='k',
            linewidth=0.3
        )

        # For clean legend
        legend_patches.append(mpatches.Patch(color=color, label=label))

    # Clean legend outside plot
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.05, 1))

    ax.set_title("Projected Controllability Ellipsoids onto XYZ")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.show()

def train_rbf():
    a = np.load('./Trajectory-Simulation/A_trainer_5_large.npy')
    b = np.load('./Trajectory-Simulation/B_trainer_5_large.npy')
    trainer_vals = np.load('./Trajectory-Simulation/sobolsequence_5_large.npy')

    iterations = len(trainer_vals[:, 0])
    a_reshape = a.reshape(iterations, -1)
    b_reshape = b.reshape(iterations, -1)

    interp_a = RBFInterpolant(trainer_vals, a_reshape)
    interp_b = RBFInterpolant(trainer_vals, b_reshape)

    return interp_a, interp_b

def calc_c(a, b):
    """
    Build controllability matrix C = [B, AB, A^2B, ..., A^(n-1)B]
    """
    n = a.shape[0]
    m = b.shape[1]
    C = np.zeros((n, n * m))
    for i in range(n):
        C[:, i*m:(i+1)*m] = np.linalg.matrix_power(a, i) @ b
    return C

# Load interpolators
A_interp, B_interp = train_rbf()

A1 = A_interp([[15, -15, 0]]).reshape(-1, 12, 12)[0]
B1 = B_interp([[15, -15, 0]]).reshape(-1, 12, 4)[0]
C1 = calc_c(A1, B1)
U1, S1, _ = np.linalg.svd(C1)

A2 = A_interp([[15, 15, 0]]).reshape(-1, 12, 12)[0]
B2 = B_interp([[15, 15, 0]]).reshape(-1, 12, 4)[0]
C2 = calc_c(A2, B2)
U2, S2, _ = np.linalg.svd(C2)
print(U2)

# ========== Third Flight Condition ==========

A3 = A_interp([[15, 0, 0]]).reshape(-1, 12, 12)[0]
B3 = B_interp([[15, 0, 0]]).reshape(-1, 12, 4)[0]
C3 = calc_c(A3, B3)
U3, S3, _ = np.linalg.svd(C3)

# Plot all 3 ellipsoids together
plot_multiple_ellipsoids(
    ellipsoids=[
        (U1, S1, "Flight Condition (15, -15, 10)", "blue"),
        (U2, S2, "Flight Condition (15, 15, 0)", "red"),
        (U3, S3, "Flight Condition (15, 0, 0)", "green"),
    ],
    idx=[3, 7, 11]
)