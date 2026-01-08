# Optimal Control and Reinforcement Learning Project 2025-2026
# Group 23: Alessandro Binci, Alessandro Tampieri, Lorenzo Tucci
# Problem 2 with set of parameters 1

import numpy as np
import sympy as sy
import math

import dynamics as dyn

import equilibrium as eq

# Defining the discretization step value
dt = dyn.dt

# Defining final time in seconds
tf = 10

# Time of transition start
t_init = 3.5
T_init = int(t_init/dt)

# Time of transition finish
t_end = 6.5
T_end = int(t_end/dt)

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

def trapezoidal(ns,ni,th1, tau_init, th1_final): 

    # Trapezoidal period

    TT_trap = T_end - T_init

    # Initializing the trapezoidal reference curve
    xx_ref = np.zeros((ns, TT_trap))
    uu_ref = np.zeros((ni, TT_trap))

    # Importing equilibrium points
    xx_eq1, xx_eq2, uu_eq1, uu_eq2 = eq.eq_gen(th1, tau_init, th1_final)

    # Initial points for trapezoidal trajectory
    xx_init = xx_eq1
    uu_init = uu_eq1

    # Final points for trapezoidal trajectory
    xx_final = xx_eq2
    uu_final = uu_eq2

    # Displacement on the state and displacement on the input

    L_xx = xx_final[0:2] - xx_init[0:2]
    L_uu = uu_final - uu_init

    # Constant velocity: v = (3*L)/(2*T) -> 1.5 * L / T, L is the displacement and T is the period

    vel_c_xx = (1.5 * L_xx) / (TT_trap * dt)
    vel_c_uu = (1.5 * L_uu) / (TT_trap * dt)

    # Max acceleration: a = (9*L)/(2*T^2) -> 4.5 * L / T^2, L is the displacement and T is the period
    acc_max_xx = (4.5 * L_xx) / ((TT_trap * dt)**2)
    acc_max_uu = (4.5 * L_uu) / ((TT_trap * dt)**2)

    for k in range(ns):

        for tt in range(TT_trap):

            if tt<=(TT_trap/3):
                xx_ref[k,tt] = xx_init[k] + 0.5*acc_c[k]*((tt*dt)^2)

            if tt > (TT_trap*(2/3)):
                xx_ref[k,tt] = xx_final[k] - 0.5*acc_c[k]*(((TT_trap-tt)*dt)^2)
                
            else:
                xx_ref[k,tt] = xx_init[k] + acc_c[k]* ((TT_trap/3)*dt)* ((tt-((TT_trap/3)/2))*dt) 

    for k in range(ni):

        for tt in range(TT_trap):

            if tt<=(TT_trap/3):
                uu_ref[k,tt] = uu_init[k] + 0.5*acc_c[k]*((tt*dt)^2)

            if tt > (TT_trap*(2/3)):
                uu_ref[k,tt] = uu_final[k] - 0.5*acc_c[k]*(((TT_trap-tt)*dt)^2)
                
            else: 
                uu_ref[k,tt] = uu_init[k] + acc_c[k]* ((TT_trap/3)*dt)* ((tt-((TT_trap/3)/2))*dt)        
     

    return xx_ref , uu_ref










def gen_smooth(tf, dt, ns, ni, th1, tau_init, th1_final):

    # Total number of steps
    TT = int(tf/dt) 

  

    # Initializing the reference curve
    xx_ref = np.zeros((ns, TT))
    uu_ref = np.zeros((ni, TT))

    # Importing equilibrium points
    xx_eq1, xx_eq2, uu_eq1, uu_eq2 = eq.eq_gen(th1, tau_init, th1_final)



