# Optimal Control and Reinforcement Learning Project 2025-2026
# Group 23: Alessandro Binci, Alessandro Tampieri, Lorenzo Tucci
# Problem 2 with set of parameters 1

import numpy as np
import casadi as ca

def solver_linear_MPC(AA_window, BB_window, QQ, RR, QQf, delta_x0, T_pred, uu_ref_window, umax, umin, ipopt_opts=None):
    r"""
	LQR for LTV system with (time-varying) affine cost
	
  Args
    - AA_window is a (ns x ns (x T_window)) linearization matrix
    - BB_window is a (ns x ni (x T_window)) linearization matrix
    - QQ is a (ns x ns (x TT)) stage cost matrix
    - RR is a (ni x ni (x TT)) stage cost matrix
    - QQf is a (ns x ns) terminal cost matrix
    - delta_x0 is the error on the state (measured state - optimal state)
    - T_pred is T_window, i.e. the time of prediction window
    - uu_ref_window is the input reference vector that has dimension equal to (ni x (x T_window))
    - umax is the upper bound on the input torque (constraint)
    - umin is the lower bound on the input torque (constraint)
    - ipopt_opts are the solver options
  Return
    - uu_pred[:,0] is the first optimal input of the optimal input sequence for that time window (returned as numpy array).
        If the solver can't find the optimal input a sequence of zeros is returned
    """
    # Setup of Casadi variables
    opti = ca.Opti()
    
    # State variables (Errors: delta_x)
    # Size: ns x (T_pred + 1) to include the initial state
    ns, ni = BB_window.shape[0], BB_window.shape[1]
    xx_pred = opti.variable(ns, T_pred + 1)
    
    # Input variables (Corrections: delta_u)
    # Size: ni x T_pred
    uu_pred = opti.variable(ni, T_pred)

    # Converting weights to Casadi 2D matrices
    QQ = ca.DM(QQ)
    RR = ca.DM(RR)
    QQf = ca.DM(QQf)
    delta_x0 = ca.DM(delta_x0)

    # Initializing the cost function
    cost = 0
    
    # Initial Condition Constraint: the simulation starts at the current error
    opti.subject_to(xx_pred[:, 0] == delta_x0)
    
    # Constraints loop
    for tt in range(T_pred):
        
        # Local variables for cleaner code
        xt = xx_pred[:, tt]   # delta_x at step tt
        ut = uu_pred[:, tt]     # delta_u at step tt
        ut_ref = uu_ref_window[:, tt] # Reference input at step tt (needed for constraints)
        
        # Cost Accumulation (
        cost += ca.mtimes([xt.T, QQ, xt]) + ca.mtimes([ut.T, RR, ut])
        
        # Dynamics Constraint (LTV)
        # Note: We use index 'tt' directly because matrices have been already sliced
        A_t = ca.DM(AA_window[:,:,tt])
        B_t = ca.DM(BB_window[:,:,tt])    
           
        # Dynamics: next state = A*x_curr + B*input
        opti.subject_to(xx_pred[:, tt + 1] == ca.mtimes(A_t, xt) + ca.mtimes(B_t, ut))
        
        # Input Constraints
        opti.subject_to(ut + ut_ref <= umax)
        opti.subject_to(ut + ut_ref >= umin)


    # Terminal cost
    xt_final = xx_pred[:, T_pred]
    cost += ca.mtimes([xt_final.T, QQf, xt_final])

    # Solve: it minimizes the cost
    opti.minimize(cost)
    
    # Solver options
    if ipopt_opts is None:
        ipopt_opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 1000
        }
    opti.solver("ipopt", ipopt_opts)

    try:
        sol = opti.solve()
        # Return the first optimal input correction (delta_u_0)
        # We need to convert it back to a numpy array
        return np.array(sol.value(uu_pred[:, 0]))
    except RuntimeError as e:
        print("Solver failed to find optimal solution")
        return np.zeros(ni) # Fallback: return zero correction