"""
Model Predictive Control (MPC) Class
"""

# Import packages:
import numpy as np
import osqp
import scipy.sparse as sp

class MPC(object):

    def __init__(self, a, b, c, f, v, w3, w4, x0, y_des, u_min, u_max, f_type, l_type="LTI"):

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
        self.l_type = l_type
        self.u_sequence = None

        # Ensure function is one of the two options:
        assert self.f_type in ['linear', 'nonlinear'], "Function type must be 'linear' or 'nonlinear'"

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

        # Ensure linearization strategy is one of the two options:
        assert self.l_type in ['LTI', 'LTV'], "Linearization type must be 'LTI' or 'LTV'"

        # If LTV, we need to define sequential system matrices:
        if self.l_type == 'LTV':
            self.A_seq = a
            self.B_seq = b
            self.C_seq = c
        else:
            self.A_seq = [self.A] * self.f
            self.B_seq = [self.B] * self.f
            self.C_seq = [self.C] * self.f

        # Track current step:
        self.curr_step = 0

        # Initialize states:
        self.states = []
        self.states.append(self.x0.reshape(-1, 1))

        # Initialize i/o:
        self.inputs = []
        self.outputs = []


    # Precompute matrices to save time later:
    def precompute_matrix(self):

        # O matrix: C[i] @ Phi[i,0] where Phi[i,0] = A[i] @ A[i-1] @ ... @ A[0]
        o_mat = np.zeros((self.f * self.r, self.n))
        for i in range(self.f):
            # Compute state transition matrix from step 0 to step i+1
            Phi = np.eye(self.n)
            for step in range(i + 1):
                Phi = self.A_seq[step] @ Phi
            o_mat[i * self.r:(i + 1) * self.r, :] = self.C_seq[i] @ Phi

        # M matrix: C[i] @ Phi[i,j+1] @ B[j] where Phi[i,j+1] = A[i] @ ... @ A[j+1]
        m_mat = np.zeros((self.f * self.r, self.v * self.m))
        for i in range(self.f):
            for j in range(min(i + 1, self.v)):
                # Compute state transition matrix from step j+1 to step i+1
                Phi = np.eye(self.n)
                for step in range(j + 1, i + 1):
                    Phi = self.A_seq[step] @ Phi
                m_mat[i * self.r:(i + 1) * self.r, j * self.m:(j + 1) * self.m] = \
                    self.C_seq[i] @ Phi @ self.B_seq[j]

        # Compute gain matrix
        gain_mat = np.linalg.inv(m_mat.T @ self.W4 @ m_mat + self.W3) @ m_mat.T @ self.W4
        return o_mat, m_mat, gain_mat

    # Propagate dynamics for controller solver (x_{k+1} = Ax_{k} + Bu_{k}):
    def prop_dyn(self, u, x, step=None):

        u = u.reshape(-1, 1)
        x = x.reshape(-1, 1)

        # Pick the matrices:
        if self.l_type == 'LTV':
            assert step is not None, "Step index must be provided for LTV systems"
            A = self.A_seq[step]
            B = self.B_seq[step]
            C = self.C_seq[step]
        else:  # LTI
            A = self.A
            B = self.B
            C = self.C

        # Propagate
        xkp1 = A @ x + B @ u
        yk = C @ x

        return xkp1, yk

    # Compute control inputs:
    def control_inputs(self, a_lin, b_lin, c_lin):

        # Check if the problem is non-linear:
        if self.f_type == 'linear':
            pass

        # Expects the linearized equations:
        elif self.f_type == 'nonlinear' and a_lin is not None and b_lin is not None and c_lin is not None:

            # Define sequential matrices:
            if self.l_type == 'LTV':
                # For LTV, expect lists of matrices
                assert isinstance(a_lin, list), "For LTV, a_lin must be a list"
                assert len(a_lin) == self.f, f"Expected {self.f} matrices, got {len(a_lin)}"

                self.A_seq = a_lin
                self.B_seq = b_lin
                self.C_seq = c_lin

                # Use first matrix for dimension extraction
                self.A = a_lin[0]
                self.B = b_lin[0]
                self.C = c_lin[0]
            else:
                # For LTI, single matrices
                self.A = a_lin
                self.B = b_lin
                self.C = c_lin

                self.A_seq = [self.A] * self.f
                self.B_seq = [self.B] * self.f
                self.C_seq = [self.C] * self.f

            # Fetch dimensions (now A, B, C are always single 2D matrices)
            self.n = self.A.shape[0]
            self.m = self.B.shape[1]
            self.r = self.C.shape[0]

            # Compute MPC matrices now that we have A and B:
            self.O, self.M, self.Gain = self.precompute_matrix()

        # --- Solve via Closed-Form Solution ---
        if self.u_max is None and self.u_min is None:
            # Extract current desired trajectory:
            # FIX: Add +1 to get reference at k+1:k+f+1 instead of k:k+f
            y_des_curr = self.y_des[self.curr_step + 1:self.curr_step + self.f + 1, :]  # shape (f, r)

            # Compute the s vector:
            s = y_des_curr.flatten() - (self.O @ self.states[self.curr_step]).flatten()  # shape (f*r,)

            # Compute the control sequence:
            input_seq = self.Gain @ s  # shape (v*m,)
            input_seq = input_seq.reshape(self.B.shape[1], -1)

            # Store the full sequence:
            self.u_sequence = input_seq

            # Apply only first input:
            u0 = np.array([[input_seq[:, 0]]])  # column vector

            # Propagate one step:
            state_kp1, output_k = self.prop_dyn(u0, self.states[self.curr_step], self.curr_step)

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
        # FIX: Add +1 to get reference at k+1:k+f+1 instead of k:k+f
        y_des_curr = self.y_des[self.curr_step + 1:self.curr_step + self.f + 1, :]
        y_des_curr = y_des_curr.flatten().reshape(-1, 1)

        # Compute prediction offset:
        Ox = (self.O @ self.states[self.curr_step]).reshape(-1, 1)
        s = Ox - y_des_curr

        # Build QP cost:
        H = self.M.T @ self.W4 @ self.M + self.W3
        H = (H + H.T) / 2

        # Cost linear term:
        f = (self.M.T @ (self.W4 @ s)).reshape(self.m * self.v)

        # Convert to sparse (needed for OSQP):
        H_sp = sp.csc_matrix(H)
        f_sp = np.array(f).flatten()

        # Build constraints for OSQP:
        umin = np.repeat(self.u_min, self.v)
        umax = np.repeat(self.u_max, self.v)

        # OSQP uses l <= A*x <= u:
        A_ineq = sp.eye(self.m * self.v).tocsc()

        # Construct bounds:
        l_bound = umin
        u_bound = umax

        # Solve QP for Control Solution:
        prob = osqp.OSQP()
        prob.setup(
            P=H_sp,
            q=f_sp,
            A=A_ineq,
            l=l_bound,
            u=u_bound,
            verbose=False
        )

        # Unpack solution:
        res = prob.solve()

        # Define controls:
        u_opt = res.x.reshape(self.v, self.m).T  # shape (m, v)
        u0 = u_opt[:, 0]

        # Store the full optimal sequence
        self.u_sequence = u_opt

        # Do not propagate if nonlinear, want to test ability to linearize and track:
        if not self.f_type == 'nonlinear':
            # Propagate dynamics:
            next_x, yk = self.prop_dyn(u0, self.states[self.curr_step], self.curr_step)

            # Log data back to driver file:
            self.states.append(next_x)
            self.outputs.append(yk)

        # Append control input:
        self.inputs.append(u0)

        # Advance step:
        self.curr_step += 1
