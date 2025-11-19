# Import packages:
import numpy as np
from rbf.interpolate import RBFInterpolant
import matplotlib.pyplot as plt

# Train RBF for A and B matrices:
def train_rbf():
    a = np.load('./AVL_Data/A_trainer_5_large.npy')
    b = np.load('./AVL_Data/B_trainer_5_large.npy')
    trainer_vals = np.load('./AVL_Data/sobolsequence_5_large.npy')

    iterations = len(trainer_vals[:, 0])
    a_reshape = a.reshape(iterations, -1)
    b_reshape = b.reshape(iterations, -1)

    interp_a = RBFInterpolant(trainer_vals, a_reshape)
    interp_b = RBFInterpolant(trainer_vals, b_reshape)

    return interp_a, interp_b

# Calculate controllability matrix:
def calc_c(a, b, steps):
    n = a.shape[0]
    m = b.shape[1]
    c = np.zeros((n, steps * m))
    # Build controllability matrix in reverse order: [A^(N-1)*B, A^(N-2)*B, ..., A*B, B]
    for i in range(steps):
        c[:, i*m:(i+1)*m] = np.linalg.matrix_power(a, steps-1-i) @ b
    return c

# Problem parameters:
x_0 = np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
x_f = np.array([15, 0, 0, 0, 0, 0, 0, 0, 100, 50, 10, 0])
n_tsteps = 100  # Reduced from 100 to avoid numerical overflow

# Load interpolators:
A_interp, B_interp = train_rbf()

# Fetch matrices:
A = A_interp([[15, 6, 0]]).reshape(-1, 12, 12)[0]
B = B_interp([[15, 6, 0]]).reshape(-1, 12, 4)[0]

print(f"System eigenvalues: {np.linalg.eigvals(A)}")
print(f"Max eigenvalue magnitude: {np.max(np.abs(np.linalg.eigvals(A))):.4f}")

C = calc_c(A, B, n_tsteps)

# Check controllability:
rank = np.linalg.matrix_rank(C)
print(f"Controllability matrix rank: {rank}/{A.shape[0]}")
if rank != A.shape[0]:
    print("WARNING: System may not be controllable")

# Solve for control inputs using pseudoinverse:
RHS = x_f - np.linalg.matrix_power(A, n_tsteps) @ x_0
U_flat = np.linalg.pinv(C) @ RHS
U = U_flat.reshape(n_tsteps, B.shape[1])

print(f"Control input shape: {U.shape}")
print(f"First control input: {U[0]}")

# Simulate:
x = np.zeros((A.shape[0], n_tsteps+1))
x[:, 0] = x_0
for k in range(n_tsteps):
    x[:, k+1] = A @ x[:, k] + (B @ U[k]).flatten()

# Verification:
print(f"\nInitial state: {x[:, 0]}")
print(f"Final state: {x[:, n_tsteps]}")
print(f"Desired final state: {x_f}")
print(f"Error: {np.linalg.norm(x[:, n_tsteps] - x_f):.6f}")

# Plot trajectory:
plt.figure(figsize=(10, 8))

# Plot x vs y
plt.subplot(2, 2, 1)
plt.plot(x[0,:], x[1,:], marker='o', markersize=3, label='Trajectory')
plt.scatter(x_0[0], x_0[1], color='green', s=100, label='Initial state', zorder=5)
plt.scatter(x_f[0], x_f[1], color='red', s=100, label='Desired final state', zorder=5)
plt.scatter(x[0, n_tsteps], x[1, n_tsteps], color='blue', s=100, marker='x', label='Actual final state', zorder=5)
plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.title('State Trajectory (x[0] vs x[1])')
plt.grid(True)
plt.legend()

# Plot all states over time
plt.subplot(2, 2, 2)
for i in range(min(6, x.shape[0])):
    plt.plot(x[i,:], label=f'x[{i}]')
plt.xlabel('Time step')
plt.ylabel('State value')
plt.title('First 6 States Over Time')
plt.grid(True)
plt.legend()

# Plot control inputs
plt.subplot(2, 2, 3)
for i in range(U.shape[1]):
    plt.plot(U[:, i], label=f'u[{i}]')
plt.xlabel('Time step')
plt.ylabel('Control input')
plt.title('Control Inputs')
plt.grid(True)
plt.legend()

# Plot error over time
plt.subplot(2, 2, 4)
errors = np.linalg.norm(x[:, :] - x_f.reshape(-1, 1), axis=0)
plt.plot(errors)
plt.xlabel('Time step')
plt.ylabel('Error to target')
plt.title('Distance to Target State')
plt.grid(True)

plt.tight_layout()
plt.show()