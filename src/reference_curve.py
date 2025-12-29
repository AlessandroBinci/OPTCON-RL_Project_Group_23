# Optimal Control and Reinforcement Learning Project 2025-2026
# Group 23: Alessandro Binci, Alessandro Tampieri, Lorenzo Tucci
# Problem 2 with set of parameters 1

import numpy as np
import sympy as sy
import math

import opt_project_group23_dyn as dyn

import opt_project_group23_equilibrium as eq

# Defining final time in seconds
tf = 10

def gen(tf, dt, ns, ni, th1, tau_init, th1_final):

    # Total number of steps
    TT = int(tf/dt) 

    # Initializing the reference curve
    xx_ref = np.zeros((ns, TT))
    uu_ref = np.zeros((ni, TT))

    # Importing equilibrium points
    xx_eq1, xx_eq2, uu_eq1, uu_eq2 = eq.eq_gen(th1, tau_init, th1_final)
   

    #Computing the reference step function
    for k in range(ns):
        xx_ref[k, 0:int(TT/2)] = xx_eq1[k]
        xx_ref[k, int(TT/2):] = xx_eq2[k]

    for k in range(ni):
        uu_ref[k, 0:int(TT/2)] = uu_eq1[k]
        uu_ref[k, int(TT/2):] = uu_eq2[k]

    return xx_ref, uu_ref