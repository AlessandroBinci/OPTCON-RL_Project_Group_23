# Optimal Control and Reinforcement Learning Project 2025-2026
# Group 23: Alessandro Binci, Alessandro Tampieri, Lorenzo Tucci
# Problem 2 with set of parameters 1

import numpy as np
import casadi as ca

def solver_linear_MPC(AA_window, BB_window, QQ, RR, QQf, delta_x0, T_pred, uu_ref_window, umax, umin, ipopt_opts=None): #eliminato xxt
    
    # Setup of Casadi variables
    opti = ca.Opti()
    
    # State variables (Errors: delta_x)
    # size: ns x (T_pred + 1) to include the initial state
    ns, ni = BB_window.shape[0], BB_window.shape[1]
    xx_pred = opti.variable(ns, T_pred + 1)
    
    # Input variables (Corrections: delta_u)
    # size: ni x T_pred
    uu_pred = opti.variable(ni, T_pred)
    
    #xx_pred = opti.variable(ns, T_pred)
    #uu_pred = opti.variable(ni, T_pred)

    # Converting weights to Casadi 2D matrices
    QQ = ca.DM(QQ)
    RR = ca.DM(RR)
    QQf = ca.DM(QQf)
    delta_x0 = ca.DM(delta_x0)
    #xxt = np.array(xxt).squeeze()

    # Cost function and constraints loop
    cost = 0
    
    # Initial Condition Constraint: The simulation starts at the current error
    opti.subject_to(xx_pred[:, 0] == delta_x0)
    
    for tt in range(T_pred):
        
        # Local variables for cleaner code
        xt = xx_pred[:, tt]   # delta_x at step tt
        ut = uu_pred[:, tt]     # delta_u at step tt
        ut_ref = uu_ref_window[:, tt] # Reference input at step tt (needed for constraints)
        
        # Cost Accumulation (Quadratic Regulation)
        cost += ca.mtimes([xt.T, QQ, xt]) + ca.mtimes([ut.T, RR, ut])
        
        # Dynamics Constraint (LTV)
        # Note: We use index 'k' directly because matrices are already sliced
        A_t = ca.DM(AA_window[:,:,tt])
        B_t = ca.DM(BB_window[:,:,tt])    
           
        # Dynamics: next state = A*curr + B*input
        opti.subject_to(xx_pred[:, tt + 1] == ca.mtimes(A_t, xt) + ca.mtimes(B_t, ut))
        
        # Input Constraints (on TOTAL input)
        # We want: u_min <= (u_ref + delta_u) <= u_max
        # So: u_min - u_ref <= delta_u <= u_max - u_ref
        
        # Note: If you want to test WITHOUT constraints first, 
        # set umax=1000 and umin=-1000 in the main file.
        opti.subject_to(ut + ut_ref <= umax)
        opti.subject_to(ut + ut_ref >= umin)


    # Terminal cost
    xt_final = xx_pred[:, T_pred]
    cost += ca.mtimes([xt_final.T, QQf, xt_final])

    # Solve
    opti.minimize(cost)
    
    # solver options
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