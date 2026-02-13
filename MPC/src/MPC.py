# Import packages:
import numpy as np
import osqp
import scipy.sparse as sp
import time


# Model Predictive Control (MPC) Class:
class MPC(object):

    # Initialize class:
    def __init__(self, a, b, c, f, v, w3, w4, x0, y_des, u_min, u_max, f_type, l_type="LTI"):

        # Unpack inputs:
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

        # Must be either of these types:
        assert self.f_type in ['linear', 'nonlinear']
        assert self.l_type in ['LTI', 'LTV']

        # If linear, precompute everything:
        if self.f_type == 'linear':
            self.n = self.A.shape[0]
            self.m = self.B.shape[1]
            self.r = self.C.shape[0]
            self.O, self.M, self.Gain = self.precompute_matrix()
            self._W4M = self.W4 @ self.M
            self._MtW4 = self.M.T @ self.W4
            self._MtW4M = self._MtW4 @ self.M

        # If not, just initialize to None:
        else:
            self.n = None
            self.m = None
            self.r = None
            self.O = None
            self.M = None
            self.Gain = None

        # If LTV, define sequential matrices:
        if self.l_type == 'LTV':
            self.A_seq = a
            self.B_seq = b
            self.C_seq = c

        # If LTI, just repeat the "stationary" matrices:
        else:
            self.A_seq = [self.A] * self.f if self.A is not None else None
            self.B_seq = [self.B] * self.f if self.B is not None else None
            self.C_seq = [self.C] * self.f if self.C is not None else None

        # Initialize everything else:
        self.curr_step = 0
        self.states = [self.x0.reshape(-1, 1)]
        self.inputs = []
        self.outputs = []
        self.x_ref_seq = None
        self.r_seq = None
        self.d_k = None

    # Precompute matrices for MPC:
    def precompute_matrix(self):

        # Unpack dimensions:
        n, m, r = self.n, self.m, self.r
        f, v = self.f, self.v

        # Build cumulative Phi:
        Phi = [np.eye(n)]
        for i in range(f):
            Phi.append(self.A_seq[i] @ Phi[-1])

        # Precompute inverses of Phi:
        Phi_inv = [np.linalg.inv(Phi_k) for Phi_k in Phi]

        # Compute O matrix:
        o_mat = np.zeros((f * r, n))
        for i in range(f):
            o_mat[i * r:(i + 1) * r, :] = self.C_seq[i] @ Phi[i + 1]

        # Compute M matrix:
        m_mat = np.zeros((f * r, v * m))
        for i in range(f):
            Ci = self.C_seq[i]
            Phi_i = Phi[i + 1]
            for j in range(min(i + 1, v)):
                Phi_j_inv = Phi_inv[j + 1]
                Phi_ij = Phi_i @ Phi_j_inv
                m_mat[i * r:(i + 1) * r, j * m:(j + 1) * m] = Ci @ Phi_ij @ self.B_seq[j]

        # Solve for gain:
        MtW4 = m_mat.T @ self.W4
        gain_mat = np.linalg.solve(MtW4 @ m_mat + self.W3, MtW4)

        # Return matrices:
        return o_mat, m_mat, gain_mat

    # Precompute r offset:
    def compute_r_offset(self):

        # Can only compute if r is defined:
        if self.r_seq is None or all(r is None for r in self.r_seq):
            return np.zeros((self.f * self.r, 1))

        # Unpack dimensions:
        n, r = self.n, self.r
        f = self.f

        # Build cumulative Phi:
        Phi = [np.eye(n)]
        for i in range(f):
            Phi.append(self.A_seq[i] @ Phi[-1])

        # Precompute inverse of Phi:
        Phi_inv = [np.linalg.inv(Phi_k) for Phi_k in Phi]

        # Initialize r_offset:
        r_offset = np.zeros((f * r, 1))

        # Loop through horizon:
        for i in range(f):

            # Solve for state contributions of Phi and r_offset:
            x_r_accum = np.zeros((n, 1))
            Phi_i = Phi[i + 1]
            for j in range(i + 1):
                if self.r_seq[j] is None:
                    continue
                Phi_j_inv = Phi_inv[j + 1]
                Phi_ij = Phi_i @ Phi_j_inv
                x_r_accum += Phi_ij @ self.r_seq[j].reshape(-1, 1)

            # Update r_offset:
            r_offset[i * r:(i + 1) * r, :] = self.C_seq[i] @ x_r_accum

        # Return r:
        return r_offset

    # Propagate dynamics:
    def prop_dyn(self, u, x, step=None):

        # Reshape to ensure correct dimensions:
        u = u.reshape(-1, 1)
        x = x.reshape(-1, 1)

        # If LTV, just pull current step:
        if self.l_type == 'LTV':
            A = self.A_seq[step]
            B = self.B_seq[step]
            C = self.C_seq[step]

        # If LTI, just pull stationary matrices:
        else:
            A = self.A
            B = self.B
            C = self.C

        # Propagate and return states:
        xkp1 = A @ x + B @ u
        yk = C @ x
        return xkp1, yk

    # Solve for current control step:
    def control_inputs(self, a_lin, b_lin, c_lin, x_ref_seq=None, r_seq=None, u_nom_seq=None):

        # Initialize timing objects:
        timing = {}
        t_start_total = time.perf_counter()

        # If linear, don't need to do any of the linearization steps:
        if self.f_type == 'linear':
            pass

        # If nonlinear, initialize:
        elif self.f_type == 'nonlinear' and a_lin is not None:

            # For LTV, apply sequential matrices:
            if self.l_type == 'LTV':
                self.A_seq = a_lin
                self.B_seq = b_lin
                self.C_seq = c_lin
                self.x_ref_seq = x_ref_seq
                self.r_seq = r_seq if r_seq is not None else [None] * self.f
                self.A = a_lin[0]
                self.B = b_lin[0]
                self.C = c_lin[0]

            # If LTI, just pull stationary matrices and repeat for sequential:
            else:
                self.A = a_lin
                self.B = b_lin
                self.C = c_lin
                self.A_seq = [self.A] * self.f
                self.B_seq = [self.B] * self.f
                self.C_seq = [self.C] * self.f
                self.r_seq = [None] * self.f

            # Unpack dimensions:
            self.n = self.A.shape[0]
            self.m = self.B.shape[1]
            self.r = self.C.shape[0]

            # Precompute matrices:
            t_start = time.perf_counter()
            if self.l_type == 'LTV' or self.O is None:
                self.O, self.M, self.Gain = self.precompute_matrix()
                self._W4M = self.W4 @ self.M
                self._MtW4 = self.M.T @ self.W4
                self._MtW4M = self._MtW4 @ self.M
            timing['Precompute Matrices'] = time.perf_counter() - t_start

        # For LTV, calculate perturbation:
        if self.l_type == 'LTV' and self.x_ref_seq is not None:
            delta_x0 = self.states[self.curr_step] - self.x_ref_seq[0].reshape(-1, 1)

        # For LTI, perturbation = absolute so doesn't matter:
        else:
            delta_x0 = self.states[self.curr_step]

        # --- CLOSED FORM SOLUTION ---
        if self.u_max is None and self.u_min is None:

            # Initialize reference trajectory:
            if self.l_type == 'LTV':
                delta_y_des = np.zeros((self.f * self.r, 1))
            else:
                y_des_curr = self.y_des[self.curr_step + 1:self.curr_step + self.f + 1, :]
                delta_y_des = y_des_curr.flatten().reshape(-1, 1)

            # Compute r_offset and s:
            r_offset = self.compute_r_offset()
            s = (self.O @ delta_x0) + r_offset - delta_y_des

            # Compute perturbed control and append:
            delta_u_seq = self.Gain @ s
            delta_u_seq = delta_u_seq.reshape(self.m, -1)
            self.u_sequence = delta_u_seq
            delta_u0 = delta_u_seq[:, 0]
            self.inputs.append(delta_u0.reshape(-1, 1))

            # Propagate dynamics:
            if self.f_type == 'linear':
                state_kp1, output_k = self.prop_dyn(delta_u0, delta_x0, self.curr_step)
                self.states.append(state_kp1)
                self.outputs.append(output_k)

            # Step, return:
            self.curr_step += 1
            timing['total'] = time.perf_counter() - t_start_total
            self._store_timing(timing)
            return

        # --- OPEN FORM SOLUTION ---
        # Fix bounds if necessary:
        if self.u_max is None:
            self.u_max = np.inf
        if self.u_min is None:
            self.u_min = -np.inf

        # Initialize reference trajectory:
        if self.l_type == 'LTV':
            delta_y_des = np.zeros((self.f * self.r, 1))
        else:
            y_des_curr = self.y_des[self.curr_step + 1:self.curr_step + self.f + 1, :]
            delta_y_des = y_des_curr.flatten().reshape(-1, 1)

        # Precompute r_offset:
        t_start = time.perf_counter()
        r_offset = self.compute_r_offset()
        timing['R offset'] = time.perf_counter() - t_start

        # Compute s from offset and calculate H and f (as well as sparse matrices):
        t_start = time.perf_counter()
        s = (self.O @ delta_x0) + r_offset - delta_y_des
        H = self._MtW4M + self.W3
        H = (H + H.T) / 2
        f = (self._MtW4 @ s).flatten()
        H_sp = sp.csc_matrix(H)
        f_sp = np.array(f).flatten()

        # Prepare bounds:
        if self.l_type == 'LTV' and u_nom_seq is not None:
            u_nom_stack = np.hstack([u_nom_seq[k] for k in range(self.v)])
            umin = np.repeat(self.u_min, self.v) - u_nom_stack
            umax = np.repeat(self.u_max, self.v) - u_nom_stack
        else:
            umin = np.repeat(self.u_min, self.v)
            umax = np.repeat(self.u_max, self.v)

        # Finish setting up matrices:
        A_ineq = sp.eye(self.m * self.v).tocsc()
        timing['Setup Optimization'] = time.perf_counter() - t_start

        # Warm start if able (i.e. previous solution from last step):
        t_start = time.perf_counter()
        if hasattr(self, '_osqp_prob') and self.l_type == 'LTI':
            self._osqp_prob.update(q=f_sp, l=umin, u=umax)
            if hasattr(self, '_last_solution'):
                self._osqp_prob.warm_start(x=self._last_solution)
        else:
            self._osqp_prob = osqp.OSQP()
            self._osqp_prob.setup(P=H_sp, q=f_sp, A=A_ineq, l=umin, u=umax, eps_abs=1e-6, eps_rel=1e-6, verbose=False)

        # Solve optimization problem:
        res = self._osqp_prob.solve()
        self._last_solution = res.x
        timing['Solve Optimization'] = time.perf_counter() - t_start

        # Unpack perturbed controls and append:
        delta_u_opt = res.x.reshape(self.v, self.m).T
        delta_u0 = delta_u_opt[:, 0]
        self.u_sequence = delta_u_opt
        self.inputs.append(delta_u0.reshape(-1, 1))

        # If linear, propagate with the linear system:
        if self.f_type == 'linear':
            next_x, yk = self.prop_dyn(delta_u0, delta_x0, self.curr_step)
            self.states.append(next_x)
            self.outputs.append(yk)

        # Step, calculate timings:
        self.curr_step += 1
        timing['total'] = time.perf_counter() - t_start_total
        self._store_timing(timing)

    # Store timing information:
    def _store_timing(self, timing):
        if not hasattr(self, 'timing_history'):
            self.timing_history = []
        self.timing_history.append(timing)

    # Pack summary:
    def get_timing_summary(self):
        if not hasattr(self, 'timing_history') or len(self.timing_history) == 0:
            return None

        all_keys = set()
        for t in self.timing_history:
            all_keys.update(t.keys())
        summary = {}
        for key in all_keys:
            values = [t.get(key, 0) for t in self.timing_history]
            summary[key] = {
                'mean': np.mean(values),
                'total': np.sum(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len([v for v in values if v > 0])
            }
        return summary