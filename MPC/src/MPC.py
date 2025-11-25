"""
Model Predictive Control (MPC) Class
"""

# Import packages:
import numpy as np

# Main Class
class MPC(object):

    def __init__(self, a, b, c, f, v, w3, w4, x0, y_des):
        self.A = a
        self.B = b
        self.C = c
        self.f = f
        self.v = v
        self.W3 = w3
        self.W4 = w4
        self.x0 = x0
        self.y_des = y_des

        # Fetch dimensions:
        self.n = self.A.shape[0]
        self.m = self.B.shape[1]
        self.r = self.C.shape[0]

        # Track current step:
        self.curr_step = 0

        # Initialize states:
        self.states = []
        self.states.append(self.x0)

        # Initialize i/o:
        self.inputs = []
        self.outputs = []

        # Precompute O, M, and Gain:
        self.O, self.M, self.Gain = self.precompute_matrix()

    # Precompute matrices to save time later:
    def precompute_matrix(self):

        # Matrix O:
        o_mat = np.zeros(shape=(self.f * self.r, self.n))
        for i in range(self.f):
            a_power = np.linalg.matrix_power(self.A, i+1)
            o_mat[i * self.r:(i + 1) * self.r, :] = self.C @ a_power

        # Matrix M:
        m_mat = np.zeros((self.f * self.r, self.v * self.m))
        for i in range(self.f):

            # From current step to end of control horizon:
            if i < self.v:
                for j in range(i + 1):
                    a_power = np.linalg.matrix_power(self.A, j)
                    m_mat[i * self.r:(i + 1) * self.r, (i - j) * self.m:(i - j + 1) * self.m] = (
                            self.C @ a_power @ self.B)

            # From control horizon to prediction horizon:
            else:
                for j in range(self.v):
                    if j == 0:
                        prev_sum = np.zeros((self.n, self.n))
                        for s in range(i - self.v + 2):
                            a_power = np.linalg.matrix_power(self.A, j)
                            prev_sum = prev_sum + a_power
                        m_mat[i * self.r:(i + 1) * self.r, (self.v - 1) * self.m:self.v * self.m] = (
                            self.C @ prev_sum @ self.B)
                    else:
                        a_power = a_power @ self.A
                        m_mat[i * self.r:(i + 1) * self.r, (self.v - 1 - j) * self.m:(self.v - j) * self.m] = (
                            self.C @ a_power @ self.B)

        # Compute gain matrix:
        gain_mat = np.linalg.inv(m_mat.T @ self.W4 @ m_mat + self.W3) @ m_mat.T @ self.W4

        # Return precomputed matrices:
        return o_mat, m_mat, gain_mat

    # Propagate dynamics for controller solver (x_{k+1} = Ax_{k} + Bu_{k}):
    def prop_dyn(self, u, x):

        # Reshape to ensure dimensions:
        u = u.reshape(-1, 1)
        x = x.reshape(-1, 1)

        # Initialize:
        xkp1 = np.zeros((self.n, 1))
        yk = np.zeros((self.r, 1))

        # Propagate:
        xkp1 = self.A @ x + self.B @ u
        yk = self.C @ x

        # Return values:
        return xkp1, yk

    # Compute control inputs:
    def control_inputs(self):

        # Extract current desired trajectory:
        y_des_curr = self.y_des[self.curr_step:self.curr_step + self.f, :]  # shape (f, r)

        # Compute the s vector:
        s = y_des_curr.flatten() - (self.O @ self.states[self.curr_step]).flatten()  # shape (f*r,)

        # Compute the control sequence:
        input_seq = self.Gain @ s  # shape (v*m,)
        input_seq = input_seq.reshape(4, -1)

        # Apply only first input:
        u0 = np.array([[input_seq[:, 0]]])  # column vector

        # Propagate one step:
        state_kp1, output_k = self.prop_dyn(u0, self.states[self.curr_step])

        # Append to logs:
        self.states.append(state_kp1)
        self.outputs.append(output_k)  # shape (r,1)
        self.inputs.append(u0)

        # Advance step:
        self.curr_step += 1
