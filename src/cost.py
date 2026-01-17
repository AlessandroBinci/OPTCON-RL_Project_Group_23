# Optimal Control and Reinforcement Learning Project 2025-2026
# Group 23: Alessandro Binci, Alessandro Tampieri, Lorenzo Tucci
# Problem 2 with set of parameters 1

import numpy as np
import dynamics as dyn

ns = dyn.ns
ni = dyn.ni

# Defining cost matrices
Q1 = 100    # w/ these values, we're giving more weight to the angles tracking
Q2 = 100
Q3 = 1
Q4 = 1

QQ = np.diag([Q1,Q2,Q3,Q4])

# ni*ni
RR = np.array([[1]])

def stage_cost(xx_t, uu_t, xx_ref_t, uu_ref_t):
    r'''
    Stage cost : quadratic cost function

    Args:
        - xx_t \in \R^4 state at time t
        - xx_ref_t \in \R^4 state reference at time t
        - uu_t \in \R^1 input at time t
        - uu_ref_t \in \R^1 input reference at time t

    Return:
        - ll is the cost at (xx_t,uu_t)
        - lx is the gradient of l wrt x, at (xx_t,uu_t)
        - lu is the gradient of l wrt u, at (xx_t,uu_t)

    '''
    # Defining the cost for tracking 
    ll = 0.5*((xx_t - xx_ref_t).T @ QQ @ (xx_t - xx_ref_t)) + 0.5*((uu_t - uu_ref_t).T @ RR @ (uu_t - uu_ref_t))
    
    # Defining the gradients
    lx = QQ @(xx_t - xx_ref_t)
    lu = RR @(uu_t - uu_ref_t)

    return ll.squeeze(), lx, lu


def termcost(xxT,xxT_ref, QQT):
    r'''
    Terminal cost : quadratic cost function

    Args:
        - xxT \in \R^4 state at time final time T
        - xxT_ref \in \R^4 state reference at final time T
        - QQT matrix QQ at final time T

    Return:
        - llT is the cost at (xxT)
        - lTx is the gradient of l wrt x, at (xxT)
        

    '''
    #Defining the cost at (xxT)
    llT = 0.5*((xxT - xxT_ref).T @ QQT @ (xxT - xxT_ref))  
    
    #Defining the gradient l_T wrt x, at (xxT)
    lTx = QQT @(xxT - xxT_ref)                             

    return llT.squeeze(), lTx







