# Import packages:
import numpy as np
import osqp
import scipy.sparse as sp

# Model Predictive Control (MPC) Class:
class MPC(object):

    # Initialize MPC object:
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
            self.A_seq = [self.A] * self.f if self.A is not None else None
            self.B_seq = [self.B] * self.f if self.B is not None else None
            self.C_seq = [self.C] * self.f if self.C is not None else None

        # Track current step:
        self.curr_step = 0

        # Initialize states:
        self.states = []
        self.states.append(self.x0.reshape(-1, 1))

        # Initialize i/o:
        self.inputs = []
        self.outputs = []

        # For LTV, store reference trajectory and affine term:
        self.x_ref_seq = None
        self.u_ref_seq = None
        self.d_seq = None

    # Precompute matrices before MPC iteration:
    def precompute_matrix(self):

        # Construct O matrix:
        o_mat = np.zeros((self.f * self.r, self.n))
        for i in range(self.f):
            Phi = np.eye(self.n)
            for step in range(i + 1):
                Phi = Phi @ self.A_seq[step]
            o_mat[i * self.r:(i + 1) * self.r, :] = self.C_seq[i] @ Phi

        # Construct M matri:
        m_mat = np.zeros((self.f * self.r, self.v * self.m))
        for i in range(self.f):
            for j in range(min(i + 1, self.v)):
                Phi = np.eye(self.n)
                for step in range(j + 1, i + 1):
                    Phi = Phi @ self.A_seq[step]
                m_mat[i * self.r:(i + 1) * self.r, j * self.m:(j + 1) * self.m] = \
                    self.C_seq[i] @ Phi @ self.B_seq[j]

        # Compute gain matrix:
        gain_mat = np.linalg.inv(m_mat.T @ self.W4 @ m_mat + self.W3) @ m_mat.T @ self.W4
        return o_mat, m_mat, gain_mat

    # Compute affine offset term for more accurate nonlinear tracking:
    def compute_affine(self):

        # If affine term is not initialized, don't account for it:
        if self.d_seq is None or all(d is None for d in self.d_seq):
            return np.zeros((self.f * self.r, 1))

        # Initialize affine_term
        d = np.zeros((self.f * self.r, 1))

        # Loop through prediction horizon:
        for i in range(self.f):

            # Initialize_state
            x_affine = np.zeros((self.n, 1))

            # Predict next steps:
            for j in range(i + 1):

                # Accumulate A matrix throughout propagation:
                A_pow = np.eye(self.n)
                for step in range(j + 1, i + 1):
                    A_pow = A_pow @ self.A_seq[step]

                # Calculate contribution of affine term to trajectory:
                if self.d_seq[j] is not None:
                    x_affine += A_pow @ self.d_seq[j].reshape(-1, 1)

            # Map to output:
            d[i * self.r:(i + 1) * self.r, :] = self.C_seq[i] @ x_affine

        # Return affine term:
        return d

    # Propagate dynamics for controller solver:
    def prop_dyn(self, u, x, step=None):

        # Reshape to prevent errors later:
        u = u.reshape(-1, 1)
        x = x.reshape(-1, 1)

        # Pick the matrices:
        if self.l_type == 'LTV':

            # Get matrices from sequence:
            A = self.A_seq[step]
            B = self.B_seq[step]
            C = self.C_seq[step]

        else:

            # Get matrices from initialization (since LTI only has one trimmed matrix):
            A = self.A
            B = self.B
            C = self.C

        # Propagate dynamics:
        xkp1 = A @ x + B @ u
        yk = C @ x

        # Return next step:
        return xkp1, yk

    # Compute control inputs:
    def control_inputs(self, a_lin, b_lin, c_lin, x_ref_seq=None, u_ref_seq=None, d_seq=None):

        # Check if the problem has been initialized as nonlinear:
        if self.f_type == 'linear':
            pass

        # Expects the linearized equations:
        elif self.f_type == 'nonlinear' and a_lin is not None and b_lin is not None and c_lin is not None:

            # Define sequential matrices:
            if self.l_type == 'LTV':

                # Store affine offset terms (can be none if you want to ignore, but shouldn't be):
                self.d_seq = d_seq if d_seq is not None else [None] * self.f

                # Store values from input into the class:
                self.A_seq = a_lin
                self.B_seq = b_lin
                self.C_seq = c_lin
                self.x_ref_seq = x_ref_seq
                self.u_ref_seq = u_ref_seq

                # Extract dimensions:
                self.A = a_lin[0]
                self.B = b_lin[0]
                self.C = c_lin[0]

            else:

                # For LTI, single matrices:
                self.A = a_lin
                self.B = b_lin
                self.C = c_lin

                # Just repeat single matrices to fill sequential and neglect affine terms:
                self.A_seq = [self.A] * self.f
                self.B_seq = [self.B] * self.f
                self.C_seq = [self.C] * self.f
                self.d_seq = [None] * self.f

            # Extract dimensions:
            self.n = self.A.shape[0]
            self.m = self.B.shape[1]
            self.r = self.C.shape[0]

            # Compute MPC matrices now that we have A and B:
            self.O, self.M, self.Gain = self.precompute_matrix()

        # --- Solve via Closed-Form Solution ---
        if self.u_max is None and self.u_min is None:

            # Extract current desired trajectory:
            y_des_curr = self.y_des[self.curr_step + 1:self.curr_step + self.f + 1, :]

            # Compute affine offset from d terms:
            d_offset = self.compute_affine()

            # Calculate s = (O*x_curr + d_offset) - y_des:
            s = (y_des_curr.flatten() - (self.O @ self.states[self.curr_step]).flatten()
                 - d_offset.flatten())

            # Compute the control sequence:
            input_seq = self.Gain @ s  # shape (v*m,)
            input_seq = input_seq.reshape(self.B.shape[1], -1)

            # Store the full sequence:
            self.u_sequence = input_seq

            # Apply only first input:
            u0 = input_seq[:, 0]

            # Propagate one step:
            state_kp1, output_k = self.prop_dyn(u0, self.states[self.curr_step], self.curr_step)

            # Append to logs:
            self.states.append(state_kp1)
            self.outputs.append(output_k)  # shape (r,1)
            self.inputs.append(u0.reshape(-1, 1))

            # Advance step:
            self.curr_step += 1
            return

        # --- Solve via Optimization (with Bounds) ---
        # If only one of the bounds is None, replace with infinite bound to continue with optimization:
        if self.u_max is None:
            self.u_max = np.inf
        if self.u_min is None:
            self.u_min = -np.inf

        # Extract desired outputs for the next f steps:
        y_des_curr = self.y_des[self.curr_step + 1:self.curr_step + self.f + 1, :]
        y_des_curr = y_des_curr.flatten().reshape(-1, 1)

        # Compute affine offset from d terms:
        d_offset = self.compute_affine()

        # Calculate s = (O*x_curr + d_offset) - y_des:
        s = (self.O @ self.states[self.curr_step]).reshape(-1, 1) + d_offset - y_des_curr

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
        u_opt = res.x.reshape(self.v, self.m).T
        u0 = u_opt[:, 0]

        # Store the full optimal sequence:
        self.u_sequence = u_opt

        # Do not propagate if nonlinear, want to test ability to linearize and track:
        if not self.f_type == 'nonlinear':

            # Propagate dynamics:
            next_x, yk = self.prop_dyn(u0, self.states[self.curr_step], self.curr_step)

            # Log data back to driver file:
            self.states.append(next_x)
            self.outputs.append(yk)

        # Append control input:
        self.inputs.append(u0.reshape(-1, 1))

        # Advance step:
        self.curr_step += 1
