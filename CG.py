# Import packages:
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_lyapunov

# Calculate controllability matrix:
def calc_c(a, b, steps):
    n = a.shape[0]
    m = b.shape[1]
    c = np.zeros((n, steps * m))
    # Build controllability matrix in reverse order: [A^(N-1)*B, A^(N-2)*B, ..., A*B, B]
    for i in range(steps):
        c[:, i*m:(i+1)*m] = np.linalg.matrix_power(a, steps-1-i) @ b
    return c

# Define desired parameters:
x_0 = np.array([0, 0])
x_f = np.array([-30, -40])
n_tsteps = 100

# Define linear system:
A = np.array([[0.5, 0.1],
              [0.0, 0.7]])
B = np.array([[1.0],
              [0.5]])

# Fetch controllability matrix:
C = calc_c(A, B, n_tsteps)

# Ensure the matrix is non-singuler:
if np.linalg.matrix_rank(C) != A.shape[0]:
    raise Exception("ERROR: Singular Matrix")

# Compute the controllability Gramian:
eigvals = np.linalg.eigvals(A)
if np.all(np.abs(eigvals) < 1):
    print("Calculating Using Lyapunov Equation...")
    W = solve_discrete_lyapunov(A, B @ B.T)
else:
    print("Calculating Using N-1 Sum...")
    n = A.shape[0]
    W = np.zeros((n, n))
    for i in range(n_tsteps):
        A_k = np.linalg.matrix_power(A, n_tsteps-1-i)
        W += A_k @ B @ B.T @ A_k.T
print("Success!")

# Solve for control sequence:
RHS = x_f - np.linalg.matrix_power(A, n_tsteps) @ x_0
U = C.T @ np.linalg.inv(W) @ RHS

# Simulate:
x = np.zeros((A.shape[0], n_tsteps+1))
x[:, 0] = x_0
for k in range(n_tsteps):
    x[:, k+1] = A @ x[:, k] + (B * U[k]).reshape(-1)

# Plot trajectory:
plt.figure(figsize=(7,5))
plt.plot(x[0,:], x[1,:], marker='o', label='Trajectory')
plt.scatter(x_0[0], x_0[1], color='green', s=100, label='Initial state', zorder=5)
plt.scatter(x_f[0], x_f[1], color='red', s=100, label='Desired final state', zorder=5)
plt.scatter(x[0, n_tsteps], x[1, n_tsteps], color='blue', s=100, marker='x', label='Actual final state', zorder=5)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()
