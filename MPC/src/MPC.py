"""
Model Predictive Control (MPC) Class
"""

# Import packages:
import numpy as np
import osqp
import scipy.sparse as sp

class MPC(object):

    def __init__(self, a, b, c, f, v, w3, w4, x0, y_des, u_min, u_max, f_type):

        # Assign global class variables:
        self.A = a
        self.B = b
        self.C = c
        self.f = f
        self.v = v
        self.W3 = w3
        self.W4 = w4
        self.x0 = x0
        self.y_des = y_des
        self.u_min = u_min
        self.u_max = u_max
        self.f_type = f_type

        # Ensure function is one of the two options:
        assert self.f_type in ['linear', 'nonlinear']

        # Precompute matrices if linear problem (don't need to linearize):
        if self.f_type == 'linear':

            # Fetch dimensions:
            self.n = self.A.shape[0]
            self.m = self.B.shape[1]
            self.r = self.C.shape[0]

            # Precompute O, M, and Gain:
            self.O, self.M, self.Gain = self.precompute_matrix()

        # If we need to linearize each step, A and B change, so cannot precompute:
        else:

            # Initialize for later:
            self.n = None
            self.m = None
            self.r = None
            self.O = None
            self.M = None
            self.Gain = None

        # Print solver option:
        if self.u_max is None and self.u_min is None:
            print("SOLVER: Closed-Form")
        else:
            print("SOLVER: Bounded Optimization")

        # Track current step:
        self.curr_step = 0

        # Initialize states:
        self.states = []
        self.states.append(self.x0)

        # Initialize i/o:
        self.inputs = []
        self.outputs = []


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
    def control_inputs(self, a_lin, b_lin, c_lin):

        # Check if the problem is non-linear.
        if self.f_type == 'linear':
            pass

        # Expects the linearized equations:
        elif self.f_type == 'nonlinear' and a_lin is not None and b_lin is not None and c_lin is not None:

            # Assign matrices:
            self.A = a_lin
            self.B = b_lin
            self.C = c_lin

            # Fetch dimensions:
            self.n = self.A.shape[0]
            self.m = self.B.shape[1]
            self.r = self.C.shape[0]

            # Compute MPC matrices now that we have A and B:
            self.O, self.M, self.Gain = self.precompute_matrix()

        # --- Solve via Closed-Form Solution ---
        if self.u_max is None and self.u_min is None:

            # Extract current desired trajectory:
            y_des_curr = self.y_des[self.curr_step:self.curr_step + self.f, :]  # shape (f, r)

            # Compute the s vector:
            s = y_des_curr.flatten() - (self.O @ self.states[self.curr_step]).flatten()  # shape (f*r,)

            # Compute the control sequence:
            input_seq = self.Gain @ s  # shape (v*m,)
            input_seq = input_seq.reshape(self.B.shape[1], -1)

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
            return

        # --- Solve via Optimization (with Bounds) ---
        # If only one of the bounds is None, replace with infinite bound to continue with optimization:
        if self.u_max is None:
            self.u_max = np.inf
        if self.u_min is None:
            self.u_min = -np.inf

        # Extract desired outputs for the next f steps
        y_des_curr = self.y_des[self.curr_step:self.curr_step + self.f, :]
        y_des_curr = y_des_curr.flatten().reshape(-1, 1)

        # Compute prediction offset:
        Ox = (self.O @ self.states[self.curr_step]).reshape(-1, 1)
        s = Ox - y_des_curr

        # Build QP cost:
        H = self.M.T @ self.W4 @ self.M + self.W3
        H = (H + H.T) / 2

        # Cost linear term
        f = (self.M.T @ (self.W4 @ s)).reshape(self.m * self.v)

        # Convert to sparse:
        H_sp = sp.csc_matrix(H)
        f_sp = np.array(f).flatten()

        # --- Build constraints for control limits ---
        umin = np.tile(self.u_min, self.v)
        umax = np.tile(self.u_max, self.v)

        # OSQP uses l <= A*x <= u
        A_ineq = sp.eye(self.m * self.v).tocsc()

        l_bound = umin
        u_bound = umax

        # ---- Solve QP for Control Solution ----
        prob = osqp.OSQP()
        prob.setup(
            P=H_sp,
            q=f_sp,
            A=A_ineq,
            l=l_bound,
            u=u_bound,
            verbose=False
        )

        res = prob.solve()
        u_opt = res.x
        u_opt = u_opt.reshape(self.m, -1)

        # Extract the first control input
        u0 = u_opt[:, 0]

        # Propagate dynamics
        next_x, yk = self.prop_dyn(u0, self.states[self.curr_step])

        # Log data
        self.states.append(next_x)
        self.outputs.append(yk)
        self.inputs.append(u0)

        # Advance step:
        self.curr_step += 1
