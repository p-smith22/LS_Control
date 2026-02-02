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

        # For LTV, store reference trajectory and residual terms:
        self.x_ref_seq = None
        self.r_seq = None
        self.d_k = None

    # Precompute matrices before MPC iteration:
    def precompute_matrix(self):

        # Construct O matrix:
        o_mat = np.zeros((self.f * self.r, self.n))
        for i in range(self.f):
            Phi = np.eye(self.n)
            for step in range(i + 1):
                Phi = self.A_seq[step] @ Phi
            o_mat[i * self.r:(i + 1) * self.r, :] = self.C_seq[i] @ Phi

        # Construct M matrix:
        m_mat = np.zeros((self.f * self.r, self.v * self.m))
        for i in range(self.f):
            for j in range(min(i + 1, self.v)):
                Phi = np.eye(self.n)
                for step in range(j + 1, i + 1):
                    Phi = self.A_seq[step] @ Phi
                m_mat[i * self.r:(i + 1) * self.r, j * self.m:(j + 1) * self.m] = \
                    self.C_seq[i] @ Phi @ self.B_seq[j]

        # Compute gain matrix:
        gain_mat = np.linalg.inv(m_mat.T @ self.W4 @ m_mat + self.W3) @ m_mat.T @ self.W4
        return o_mat, m_mat, gain_mat

    # Compute affine offset from r_k terms:
    def compute_r_offset(self):

        # If r_seq is not initialized, return zeros:
        if self.r_seq is None or all(r is None for r in self.r_seq):
            return np.zeros((self.f * self.r, 1))

        # Initialize offset term:
        r_offset = np.zeros((self.f * self.r, 1))

        # Loop through prediction horizon:
        for i in range(self.f):

            # Initialize accumulated state offset from r terms:
            x_r_accum = np.zeros((self.n, 1))

            # Accumulate effect of all previous r_j on state at step i:
            for j in range(i + 1):

                # Compute product:
                A_pow = np.eye(self.n)
                for step in range(j + 1, i + 1):
                    A_pow = self.A_seq[step] @ A_pow

                # Add contribution of r_j:
                if self.r_seq[j] is not None:
                    x_r_accum += A_pow @ self.r_seq[j].reshape(-1, 1)

            # Map accumulated state offset to output:
            r_offset[i * self.r:(i + 1) * self.r, :] = self.C_seq[i] @ x_r_accum

        # Return offset term:
        return r_offset

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

    # Compute control inputs (main MPC solver):
    def control_inputs(self, a_lin, b_lin, c_lin, x_ref_seq=None, r_seq=None):

        """
        Solve MPC in PERTURBATION coordinates.

        For LTV: Dynamics are delta_x_{k+1} = A_k * delta_x_k + B_k * delta_u_k + r_k

        Args:
            a_lin: A matrices (single matrix for LTI, list for LTV)
            b_lin: B matrices (single matrix for LTI, list for LTV)
            c_lin: C matrices (single matrix for LTI, list for LTV)
            x_ref_seq: Reference state trajectory (only for LTV)
            r_seq: Linearization residual r_k = f(x_ref, u_nom) - x_ref_{k+1} (only for LTV)
        """

        # Check if the problem has been initialized as nonlinear:
        if self.f_type == 'linear':
            pass

        # Expects the linearized equations:
        elif self.f_type == 'nonlinear' and a_lin is not None and b_lin is not None and c_lin is not None:

            # Define sequential matrices:
            if self.l_type == 'LTV':

                # Store values from input into the class:
                self.A_seq = a_lin
                self.B_seq = b_lin
                self.C_seq = c_lin
                self.x_ref_seq = x_ref_seq
                self.r_seq = r_seq if r_seq is not None else [None] * self.f

                # Extract dimensions from first matrix:
                self.A = a_lin[0]
                self.B = b_lin[0]
                self.C = c_lin[0]

            else:

                # For LTI, single matrices:
                self.A = a_lin
                self.B = b_lin
                self.C = c_lin

                # Just repeat single matrices to fill sequential:
                self.A_seq = [self.A] * self.f
                self.B_seq = [self.B] * self.f
                self.C_seq = [self.C] * self.f
                self.r_seq = [None] * self.f

            # Extract dimensions:
            self.n = self.A.shape[0]
            self.m = self.B.shape[1]
            self.r = self.C.shape[0]

            # Compute MPC matrices now that we have A and B:
            self.O, self.M, self.Gain = self.precompute_matrix()

        # --- Compute initial perturbation ---
        if self.l_type == 'LTV' and self.x_ref_seq is not None:
            # delta_x_0 = x_current - x_ref_0
            delta_x0 = self.states[self.curr_step] - self.x_ref_seq[0].reshape(-1, 1)
        else:
            # For LTI or when no reference provided, use absolute state:
            delta_x0 = self.states[self.curr_step]

        # --- Solve via Closed-Form Solution (no constraints) ---
        if self.u_max is None and self.u_min is None:

            # Desired output perturbations (want delta_y = 0, i.e., track reference):
            if self.l_type == 'LTV':
                delta_y_des = np.zeros((self.f * self.r, 1))
            else:
                # For LTI, use absolute desired outputs:
                y_des_curr = self.y_des[self.curr_step + 1:self.curr_step + self.f + 1, :]
                delta_y_des = y_des_curr.flatten().reshape(-1, 1)

            # Compute offset from r_k terms:
            r_offset = self.compute_r_offset()

            # Calculate tracking error: s = O*delta_x0 + r_offset - delta_y_des
            s = (self.O @ delta_x0) + r_offset - delta_y_des

            # Compute the control perturbation sequence:
            delta_u_seq = self.Gain @ s  # shape (v*m, 1)
            delta_u_seq = delta_u_seq.reshape(self.m, -1)  # shape (m, v)

            # Store the full sequence:
            self.u_sequence = delta_u_seq

            # Apply only first control perturbation:
            delta_u0 = delta_u_seq[:, 0]

            # Store control perturbation:
            self.inputs.append(delta_u0.reshape(-1, 1))

            # Propagate one step (only for linear problems):
            if self.f_type == 'linear':
                state_kp1, output_k = self.prop_dyn(delta_u0, delta_x0, self.curr_step)
                self.states.append(state_kp1)
                self.outputs.append(output_k)

            # Advance step:
            self.curr_step += 1
            return

        # --- Solve via Optimization (with Bounds) ---
        # If only one of the bounds is None, replace with infinite bound:
        if self.u_max is None:
            self.u_max = np.inf
        if self.u_min is None:
            self.u_min = -np.inf

        # Desired output perturbations:
        if self.l_type == 'LTV':
            delta_y_des = np.zeros((self.f * self.r, 1))
        else:
            # For LTI, use absolute desired outputs:
            y_des_curr = self.y_des[self.curr_step + 1:self.curr_step + self.f + 1, :]
            delta_y_des = y_des_curr.flatten().reshape(-1, 1)

        # Compute offset from r_k terms:
        r_offset = self.compute_r_offset()

        # Calculate tracking error:
        s = (self.O @ delta_x0) + r_offset - delta_y_des

        # Build QP cost:
        H = self.M.T @ self.W4 @ self.M + self.W3
        H = (H + H.T) / 2

        # Cost linear term:
        f = (self.M.T @ self.W4 @ s).flatten()

        # Convert to sparse (needed for OSQP):
        H_sp = sp.csc_matrix(H)
        f_sp = np.array(f).flatten()

        # Build constraints for OSQP (bounds on delta_u for LTV, absolute u for LTI):
        if self.l_type == 'LTV':

            # Constraints for delta_u:
            umin = np.repeat(self.u_min, self.v)
            umax = np.repeat(self.u_max, self.v)

        else:

            # For LTI, bounds are on absolute control:
            umin = np.repeat(self.u_min, self.v)
            umax = np.repeat(self.u_max, self.v)

        # OSQP uses l <= A*x <= u:
        A_ineq = sp.eye(self.m * self.v).tocsc()

        # Solve QP for Control Solution:
        prob = osqp.OSQP()
        prob.setup(
            P=H_sp,
            q=f_sp,
            A=A_ineq,
            l=umin,
            u=umax,
            verbose=False
        )

        # Unpack solution:
        res = prob.solve()

        # Define control perturbations:
        delta_u_opt = res.x.reshape(self.v, self.m).T
        delta_u0 = delta_u_opt[:, 0]

        # Store the full optimal sequence:
        self.u_sequence = delta_u_opt

        # Store control perturbation:
        self.inputs.append(delta_u0.reshape(-1, 1))

        # Do not propagate if nonlinear, want to test ability to linearize and track:
        if self.f_type == 'linear':

            # Propagate dynamics:
            next_x, yk = self.prop_dyn(delta_u0, delta_x0, self.curr_step)

            # Log data back to driver file:
            self.states.append(next_x)
            self.outputs.append(yk)

        # Advance step:
        self.curr_step += 1
