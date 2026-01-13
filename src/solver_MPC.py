# Optimal Control and Reinforcement Learning Project 2025-2026
# Group 23: Alessandro Binci, Alessandro Tampieri, Lorenzo Tucci
# Problem 2 with set of parameters 1

import numpy as np
import casadi as ca

def solver_linear_MPC(AA, BB, QQ, RR, QQf, xxt, T_pred, ipopt_opts=None):

    # Converting the 2D matrices
    QQ = ca.DM(QQ)
    RR = ca.DM(RR)
    QQf = ca.DM(QQf)
    xxt = np.array(xxt).squeeze()


    ns, ni = BB.shape[0], BB.shape[1]

    opti = ca.Opti()
    xx_pred = opti.variable(ns, T_pred)
    uu_pred = opti.variable(ni, T_pred)

    cost = 0
    for tt in range(T_pred - 1):
        xtux = xx_pred[:, tt]
        ut = uu_pred[:, tt]
        cost += ca.mtimes([xtux.T, QQ, xtux]) + ca.mtimes([ut.T, RR, ut])
        A_t = ca.DM(AA[:,:,tt])
        B_t = ca.DM(BB[:,:,tt])       
        # dynamics
        opti.subject_to(xx_pred[:, tt + 1] == A_t @ xtux + B_t @ ut)

        #opti.subject_to(ut <= umax)
        #opti.subject_to(ut >= umin)
        # state bounds on first two states (assuming ns >= 2)
        #if ns >= 1:
            #opti.subject_to(X[0, tt] <= x1_max)
            #opti.subject_to(X[0, tt] >= x1_min)
        #if ns >= 2:
            #opti.subject_to(X[1, tt] <= x2_max)
            #opti.subject_to(X[1, tt] >= x2_min)

    # terminal cost and initial condition
    cost += ca.mtimes([xx_pred[:, T_pred - 1].T, QQf, xx_pred[:, T_pred - 1]])
    opti.subject_to(xx_pred[:, 0] == ca.DM(xxt))

    opti.minimize(cost)
    
    # solver options
    if ipopt_opts is None:
        ipopt_opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 1000}
    opti.solver("ipopt", ipopt_opts)

    try:
        sol = opti.solve()
        U0 = sol.value(uu_pred[:, 0])
        X_opt = sol.value(xx_pred)
        U_opt = sol.value(uu_pred)
        return np.asarray(U0), np.asarray(X_opt), np.asarray(U_opt)
        
    except RuntimeError as e:
        # Fallback in caso di fallimento
        print("solver_linear_mpc: solver failed")
        return np.zeros(ni), np.zeros((ns, T_pred)), np.zeros((ni, T_pred))




