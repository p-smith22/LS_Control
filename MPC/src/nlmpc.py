# Import packages:
import casadi as ca
import numpy as np
from src.MPC import MPC
from src.utils import discretize_system

# === FUNCTIONS ===
# Compute state derivatives for CasADi:
def f_nl(x, u, c):

    # Compute state derivatives and return:
    px, py, vx, vy = x[0], x[1], x[2], x[3]
    ux, uy = u[0], u[1]
    sqrt_term = ca.sqrt(vx**2 + vy**2)
    sqrt_term += 1e-4
    return ca.vertcat(vx, vy, ux - c * vx * sqrt_term, uy - c * vy * sqrt_term)

# Build nonlinear MPC problem:
def build_nlmpc(c, dt, N, v, C_mat, nx=4, nu=2, ny=4, umin=None, umax=None, W3=None, W4=None):

    # Define symbols for problem:
    x = ca.SX.sym('x', nx)
    u = ca.SX.sym('u', nu)
    du = ca.SX.sym('DU', nu, v)

    # RK4 integration for propagation:
    k1 = f_nl(x, u, c)
    k2 = f_nl(x + dt/2 * k1, u, c)
    k3 = f_nl(x + dt/2 * k2, u, c)
    k4 = f_nl(x + dt * k3, u, c)
    x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    f_disc = ca.Function('f_disc', [x, u], [x_next])

    # Parameters:
    x0_param = ca.SX.sym('X0', nx)
    yref_param = ca.SX.sym('Yref', ny, N)

    # Build controls:
    u_vals = ca.SX.zeros(nu, N)
    for i in range(N):
        for j in range(min(i+1, v)):
            u_vals[:, i] += du[:, j]

    # Simulate dynamics:
    x_vals = ca.SX.zeros(nx, N+1)
    x_vals[:, 0] = x0_param
    for i in range(N):
        x_vals[:, i+1] = f_disc(x_vals[:, i], u_vals[:, i])

    # Extract outputs:
    y_vals = ca.SX.zeros(ny, N)
    for i in range(N):
        y_vals[:, i] = ca.mtimes(C_mat, x_vals[:, i+1])

    # Reshape:
    y_vals = ca.reshape(y_vals, ny*N, 1)
    yref_param = ca.reshape(yref_param, ny*N, 1)
    du = ca.reshape(du, nu*v, 1)

    # Calculate cost:
    e_y = y_vals - yref_param
    J = ca.mtimes([e_y.T, W4, e_y]) + ca.mtimes([du.T, W3, du])

    # Control bounds:
    U_vec = ca.reshape(u_vals, nu*N, 1)

    # NLP formulation:
    opt_vars = du
    params = ca.vertcat(x0_param, yref_param)
    nlp = {
        'x': opt_vars,
        'f': J,
        'g': U_vec,
        'p': params
    }
    opts = {
        'ipopt.print_level': 0,
        'print_time': False,
        'ipopt.max_iter': 500,
        'ipopt.tol': 1e-6,
        'ipopt.acceptable_tol': 1e-4,
        'ipopt.warm_start_init_point': 'yes',
        'ipopt.mu_strategy': 'adaptive'
    }

    # Set solver:
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Bounds:
    lbx = [-ca.inf] * (nu*v)
    ubx = [ca.inf] * (nu*v)
    lbg = np.tile(umin, N)
    ubg = np.tile(umax, N)

    # Return built problem:
    return solver, lbx, ubx, lbg, ubg

# Solve the constructed MPC problem:
def solver(solver, lbx, ubx, lbg, ubg, x0, yref, nu, v, prev_du=None):

    # Warm start (shift previous step to make it faster):
    if prev_du is None:
        init_guess = np.zeros(nu*v)
    else:
        du_prev = prev_du.reshape((v, nu))
        du_new = np.vstack([du_prev[1:], np.zeros((1, nu))])
        init_guess = du_new.flatten()

    # Build parameter vector:
    params = np.concatenate([x0, yref.flatten(order='F')])

    # Solve problem:
    sol = solver(x0=init_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=params)
    du_opt = sol['x'].full().flatten()

    # Take first control:
    u0 = du_opt[:nu]

    # Return:
    return u0, du_opt
