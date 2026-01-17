# Optimal Control and Reinforcement Learning Project 2025-2026
# Group 23: Alessandro Binci, Alessandro Tampieri, Lorenzo Tucci
# Problem 2 with set of parameters 1

import numpy as np
import sympy as sy
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import dynamics as dyn

import equilibrium as eq

# Defining the discretization step value
dt = dyn.dt

# Defining final time in seconds
#tf = 10
tf = 20

# Time of transition start
#t_init = 3.5
t_init = 8.5    # in seconds
T_init = int(t_init/dt)

# Time of transition finish
#t_end = 6.5
t_end = 11.5   # in seconds
T_end = int(t_end/dt)

def gen(tf, dt, ns, ni, th1, tau_init, th1_final):

    """
        Generation of reference trajectory as step function from (x_eq1, u_eq1) to (x_eq2, u_eq2)

        Args
            - tf is the final time in seconds
            - dt is the discretization step value
            - ns is the number of states
            - ni is the number of inputs

            Args to compute equilibria:
                |- th1 is the initial angular position theta 1
                |- tau_init is the desired input torque for the equilibrium point 1
                |- th1_final is the desired angular position theta 1, for equilibrium point 2
    
        Return
        - xx_ref is the state step function reference trajectory
        - uu_ref is the input step function reference trajectory
    """

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

    """
        Generation of a (T_end - T_init)s trapezoidal trajectory w/ (x_eq1, u_eq1) as initial point
        and (x_eq2, u_eq2) as final point

        Args
            - ns is the number of states
            - ni is the number of inputs

            Args to compute equilibria:
                |- th1 is the initial angular position theta 1
                |- tau_init is the desired input torque for the equilibrium point 1
                |- th1_final is the desired angular position theta 1, for equilibrium point 2
    
        Return
        - xx_ref is the state trapezoidal trajectory
        - uu_ref is the input trapezoidal trajectory
    """

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

    # Max acceleration: a = (9*L)/(2*T^2) -> 4.5 * L / T^2, L is the displacement and T is the period
    acc_max_xx = (4.5 * L_xx) / ((TT_trap * dt)**2)
    acc_max_uu = (4.5 * L_uu) / ((TT_trap * dt)**2)

    # Time instants where the trajectory changes (1/3 and 2/3 of TT_trap)
    t1_change = int(TT_trap/3)
    t2_change = int(TT_trap * 2/3)


    # Values of position and input at the end of the first part. We'll use them as initial points for the second part
    t1_sec = t1_change * dt
    xx_t1 = xx_init[0:2] + 0.5 * acc_max_xx * (t1_sec**2)
    uu_t1 = uu_init + 0.5 * acc_max_uu * (t1_sec**2)

    # Defining the three parts of the trapezoidal trajectory
    for tt in range(TT_trap):

        # The motion in the first part is of the uniformly accelerated type 
        if tt<=(t1_change):

            xx_ref[0:2,tt] = xx_init[0:2] + 0.5*acc_max_xx[0:2]*((tt*dt)**2)
            uu_ref[:,tt] = uu_init + 0.5*acc_max_uu*((tt*dt)**2)
            xx_ref[2:4,tt] = acc_max_xx * (tt*dt)

        # The motion in the third part is of a uniformly decelerated type
        elif tt > (t2_change):

            xx_ref[0:2,tt] = xx_final[0:2] - 0.5*acc_max_xx[0:2]*(((TT_trap-tt)*dt)**2)
            uu_ref[:,tt] = uu_final - 0.5*acc_max_uu*(((TT_trap-tt)*dt)**2)
            xx_ref[2:4,tt] = acc_max_xx * ((TT_trap-tt)*dt)

        # The motion in the second part is of the uniform rectilinear type        
        else:
            xx_ref[0:2,tt] = xx_t1 + acc_max_xx[0:2]* ((t1_change)*dt)* ((tt-t1_change)*dt) 
            uu_ref[:,tt] = uu_t1 + acc_max_uu* ((t1_change)*dt)* ((tt-t1_change)*dt)
            xx_ref[2:4,tt] = vel_c_xx


    return xx_ref , uu_ref



def gen_smooth(tf, dt, ns, ni, th1, tau_init, th1_final):

    """
        Generation of reference trajectory from (x_eq1, u_eq1) to (x_eq2, u_eq2).
        The smooth transition between the two equilibria is made by using a trapezoidal trajectory

        Args
            - tf is the final time in seconds
            - dt is the discretization step value
            - ns is the number of states
            - ni is the number of inputs

            Args to compute equilibria:
                |- th1 is the initial angular position theta 1
                |- tau_init is the desired input torque for the equilibrium point 1
                |- th1_final is the desired angular position theta 1, for equilibrium point 2
    
        Return
        - xx_ref is the state reference trajectory w/ smooth transition (trapezoidal)
        - uu_ref is the input reference trajectory w/ smooth transition (trapezoidal)
    """

    # Total number of steps
    TT = int(tf/dt)

    # Initializing the reference curve
    xx_ref = np.zeros((ns, TT))
    uu_ref = np.zeros((ni, TT))

    # Importing equilibrium points
    xx_eq1, xx_eq2, uu_eq1, uu_eq2 = eq.eq_gen(th1, tau_init, th1_final)

    # Importing trapezoidal trajectory
    xx_trap_ref,uu_trap_ref = trapezoidal(ns,ni,th1, tau_init, th1_final)

    
    for tt in range(TT):

        # Defining the first part of the reference trajectory (constant at equilibrium one) 
        if tt < T_init:

            for k in range(ns):
                xx_ref[k,tt] = xx_eq1[k]

            for k in range(ni):
                uu_ref[k,tt] = uu_eq1[k]

        # Defining the last part of the reference trajectory (constant at equilibrium two)
        elif tt >= T_end:

            for k in range(ns):
                xx_ref[k,tt] = xx_eq2[k]

            for k in range(ni):
                uu_ref[k,tt] = uu_eq2[k]

        # Defining trapezoidal part        
        else:
            for k in range(ns):
                xx_ref[k,tt] = xx_trap_ref[k,tt-T_init] # 'tt-T_init' used to scale the index of xx_trap_ref

            for k in range(ni):
                uu_ref[k,tt] = uu_trap_ref[k,tt-T_init] # 'tt-T_init' used to scale the index of uu_trap_ref
    
    return xx_ref, uu_ref
                