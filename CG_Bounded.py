# Import packages:
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Calculate controllability matrix:
def calc_c(A, B, N):
    n, m = B.shape
    C = np.zeros((n, N*m))
    for i in range(N):
        C[:, i*m:(i+1)*m] = np.linalg.matrix_power(A, N-1-i) @ B
    return C

# Define desired parameters:
x0 = np.array([[0], [0]])
xf = np.array([[-30], [-20]])
dt = 0.1
time = 5
N = int(time/dt)

# Define linear system:
A = np.array([[1, 1],
              [0, 1]])
B = np.array([[0],
              [1]])

# Fetch controllability matrix:
C = calc_c(A, B, N)

# Define bounds for control:
u_min = -1
u_max = 1

# Set-up optimization problem:
U = cp.Variable((N,1))
target = xf - np.linalg.matrix_power(A, N) @ x0
constraints = [U >= u_min, U <= u_max, C @ U == target]
objective = cp.Minimize(cp.sum_squares(U))

# Solve problem:
prob = cp.Problem(objective, constraints)
prob.solve()

# Simulate:
x = np.zeros((2, N+1))
x[:, 0] = x0.flatten()
for k in range(N):
    x[:, k+1] = A @ x[:, k] + (B @ np.array([[U.value[k,0]]])).flatten()

# Plot trajectory:
plt.figure(figsize=(7,5))
plt.plot(x[0,:], x[1,:], marker='o', label='Trajectory')
plt.scatter(x0[0], x0[1], color='green', s=100, label='Initial state', zorder=5)
plt.scatter(xf[0], xf[1], color='red', s=100, label='Desired final state', zorder=5)
plt.scatter(x[0, N], x[1, N], color='blue', s=100, marker='x', label='Actual final state', zorder=5)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()
