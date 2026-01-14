# Optimal Control and Reinforcement Learning Project 2025-2026
# Group 23: Alessandro Binci, Alessandro Tampieri, Lorenzo Tucci
# Problem 2 with set of parameters 1

import numpy as np
import casadi as ca

def solver_lin_mpc(AA, BB, QQ, RR, QQf, xxt, T_pred, ipopt_opts=None):

    # Converting the 2D matrices
    QQ = ca.DM(QQ)
    RR = ca.DM(RR)
    QQf = ca.DM(QQf)

    # Assicuriamoci che lo stato iniziale sia un vettore piatto
    delta_x_init = np.array(xxt).reshape(-1)

    ns, ni = BB.shape[0], BB.shape[1]

    opti = ca.Opti()
    d_xx = opti.variable(ns, T_pred)
    d_uu = opti.variable(ni, T_pred)

    cost = 0
    for tt in range(T_pred-1):
        dx_t = d_xx[:, tt]
        du_t = d_uu[:, tt]
        cost += ca.mtimes([dx_t.T, QQ, dx_t]) + ca.mtimes([du_t.T, RR, du_t])
        A_t = ca.DM(AA[:,:,tt])
        B_t = ca.DM(BB[:,:,tt])       
        # dynamics
        opti.subject_to(d_xx[:, tt + 1] == A_t @ dx_t + B_t @ du_t)

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
    cost += ca.mtimes([d_xx[:, T_pred - 1].T, QQf, d_xx[:, T_pred - 1]])
    opti.subject_to(d_xx[:, 0] == ca.DM(xxt))

    opti.minimize(cost)
    
    # solver options
    if ipopt_opts is None:
        ipopt_opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 1000}
    opti.solver("ipopt", ipopt_opts)

    try:
        sol = opti.solve()
    
        return sol.value(d_uu[:,0])
        
    except RuntimeError:
        # Fallback in caso di fallimento
        print("solver_linear_mpc: solver failed")
        return np.zeros(ni)




