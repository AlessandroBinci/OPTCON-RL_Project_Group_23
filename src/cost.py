# Optimal Control and Reinforcement Learning Project 2025-2026
# Group 23: Alessandro Binci, Alessandro Tampieri, Lorenzo Tucci
# Problem 2 with set of parameters 1

import numpy as np
import sympy as sy
import math
import matplotlib.pyplot as plt
import control as ctrl

import dynamics as dyn

ns = dyn.ns
ni = dyn.ni

# Defining cost matrices
Q1 = 100    # w/ these values, we're giving more weught to the angles tracking
Q2 = 100
Q3 = 0.1
Q4 = 0.1

QQ = np.diag([Q1,Q2,Q3,Q4])

RR = 0.1

def stage_cost(xx_t, uu_t, xx_ref_t, uu_ref_t):
    '''
    Stage cost : quadratic cost function

    Args:
        - xx_t \in \R^4 state at time t
        - xx_ref_t \in \R^4 state reference at time t
        - uu_t \in \R^1 input at time t
        - uu_ref_t \in \R^1 input reference at time t

    Return:
        - cost at (xx_t,uu_t)
        - gradient of l wrt x, at (xx_t,uu_t)
        - gradient of l wrt u, at (xx_t,uu_t)

    '''

    ll = 0.5*((xx_t - xx_ref_t).T @ QQ @ (xx_t - xx_ref_t)) + 0.5*((uu_t - uu_ref_t).T @ RR @ (uu_t - uu_ref_t))

    lx = QQ @(xx_t - xx_ref_t)
    lu = RR @(uu_t - uu_ref_t)

    return ll.squeeze(), lx, lu


def termcost(xxT,xxT_ref, QQT):
    '''
    Terminal cost : quadratic cost function

    Args:
        - xxT \in \R^4 state at time final time T
        - xxT_ref \in \R^4 state reference at final time T
        - QQT matrix QQ at final time T

    Return:
        - cost at (xxT)
        - gradient of l wrt x, at (xxT)
        

    '''

    llT = 0.5*((xxT - xxT_ref).T @ QQT @ (xxT - xxT_ref))
    lTx = QQT @(xxT - xxT_ref)

    return llT.squeeze(), lTx







