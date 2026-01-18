# Optimal Control and Reinforcement Learning Project 2025-2026
# Group 23: Alessandro Binci, Alessandro Tampieri, Lorenzo Tucci
# Problem 2 with set of parameters 1

import numpy as np
import dynamics as dyn



def eq_gen(th1, tau_init, th1_final):

    r"""
        Computation of equilibrium configurations (x_eq1, u_eq1) and (x_eq2, u_eq2)

        Args
        - th1 is the initial angular position theta 1
        - tau_init is the desired input torque for the equilibrium point 1
        - th1_final is the desired angular position theta 1, for equilibrium point 2
    
        Return
        - x_eq1 is the state at equilbrium point 1
        - x_eq2 is the state at equilbrium point 2
        - u_eq1 is the input at equilibrium point 1 (tau_init)
        - u_eq2 is the input at equilibrium point 2
    """
    
    #------ Finding the first equilibrium--------

    
    max_iters = 500 # maximum number of iterations
    tol = 1e-8 #tolerance for the residual in the newton's method
    
    xx_curr = np.zeros(dyn.ns) # initial guess for the state
    xx_curr[0] = th1

    u_eq1 = np.zeros(dyn.ni) # initial guess for the input, that's also the first equilibrium for the input 
    u_eq1[0] = tau_init
    

    for kk in range(max_iters-1):
    
        xx_next, dfx, dfu = dyn.dynamics_euler(xx_curr,u_eq1)

        error = xx_curr - xx_next #residual function, i.e. x_k - x_k+1----> problem: min error(x_k) 

        if np.linalg.norm(error) < tol : # checking if the residual is null, so if the current state is an equilibrium

            x_eq1 = xx_curr  

        else:

            direction = np.eye(dyn.ns) - (dfx.T)

            xx_curr =  xx_curr - (np.linalg.solve(direction,error)) # Newton's update



    #------ Finding the second equilibrium--------

    xx_des = np.zeros(dyn.ns)

    xx_des[0] = th1_final
    xx_des[1] = -th1_final

    def inv_dyn_tau (th1,th2):

        r"""
            Desired torque for equilibrium point 2

            Args
            - th1 is the desired angular position theta 1
            - th2 is the desired angular position theta 2
    
            Return
            - the needed torque tau to mantain such desired position
        """
    
        tau = (dyn.g*dyn.lc_1*dyn.m_1*(np.sin(th1)))+ (dyn.g*dyn.m_2*(dyn.l_1*(np.sin(th1))+dyn.lc_2*(np.sin(th1+th2))))

        return tau


    u_eq2 = np.zeros(dyn.ni)
    tau_des = inv_dyn_tau(xx_des[0], xx_des[1])
    u_eq2[0] = tau_des

    x_eq2 = xx_des

    print(f"The first equilibrium point (x_eq1, u_eq2) is:\n{x_eq1}, {u_eq1}")
    print(f"The second equilibrium point (x_eq1, u_eq2) is:\n{x_eq2}, {u_eq2}")

    return x_eq1, x_eq2, u_eq1, u_eq2
