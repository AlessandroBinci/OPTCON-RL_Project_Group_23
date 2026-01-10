# Optimal Control and Reinforcement Learning Project 2025-2026
# Group 23: Alessandro Binci, Alessandro Tampieri, Lorenzo Tucci
# Problem 2 with set of parameters 1

import numpy as np
import sympy as sy

ns = 4
ni = 1

dt = 1e-2 # discretization stepsize - Forward Euler

# Dynamics parameters

m_1 = 1 # mass m1
m_2 = 1 # mass m2
l_1 = 1 # length of the link 1
l_2 = 1 # length of the link 2
lc_1 = 0.5 # distance between the pivot point of the link 1 and its center of mass
lc_2 = 0.5 # distance between the pivot point of the link 2 and its center of mass
inertia_1 = 0.33 # moment of inertia of the link 1 about its center of mass
inertia_2 = 0.33 # # moment of inertia of the link 2 about its center of mass
g = 9.81 # gravity
f_1 = 1 # viscous friction coefficient 1
f_2 = 1 # viscous friction coefficient 2
k = 0.5 # constant that scales the non-linear stiffness interaction modeling the elastic behavior of the link

x_sym = sy.symbols('x0:4') # State vector: x0=th1, x1=th2, x2=dth1, x3=dth2 ----ricapire'x0:4'----
u_sym = sy.symbols('u0:1') # Input vector: u0=tau 

# Extraction of the state input variables
th1, th2 = x_sym[0], x_sym[1]
dth1, dth2 = x_sym[2], x_sym[3]
tau = u_sym[0]

# Matrices
MM = sy.zeros(2,2)
MM[0,0] = inertia_1 + inertia_2 + (lc_1*m_1) + m_2*(((l_1)**2) + (2*l_1*l_2* sy.cos(th2))+ (lc_2)**2 )
MM[1,0] = inertia_2 + (lc_2*m_2)*((l_1*sy.cos(th2))+lc_2)
MM[0,1] = MM [1,0]
MM[1,1] = inertia_2 + (((lc_2)**2)*m_2)

CC = sy.zeros(2,2)
CC[0,0] = -l_1*lc_2*m_2*dth2*(sy.sin(th2))
CC[1,0] = l_1*lc_2*m_2*dth1*(sy.sin(th2))
CC[0,1] = -l_1*lc_2*m_2*(dth1+dth2)*(sy.sin(th2))
CC[1,1] = 0

GG = sy.zeros(2,1)
GG[0,0] = (g*lc_1*m_1*(sy.sin(th1)))+ (g*m_2*(l_1*(sy.sin(th1))+lc_2*(sy.sin(th1+th2))))
GG[1,0] = (g*m_2*lc_2*(sy.sin(th1+th2)))+ (3*k*(sy.sin(th2))*((sy.cos(th2))**2))

FF = sy.zeros(2,2)
FF[0,0] = f_1
FF[1,1] = f_2

acc_vec = sy.zeros(2,1) # vector that contains accelerations of th1 and th2
MM_inv = MM.inv() # computing the inverse of mass matrix
vel_vec = sy.Matrix([[dth1], [dth2]]) # it corresponds to np.array() in numpy
u_vec = sy.Matrix([[tau], [0]])
acc_vec = MM_inv @ (u_vec - GG - (CC @ vel_vec) - (FF @ vel_vec))

# Euler discretization
# x_next = x + dt * f(x,u)

dyn_cont = sy.Matrix([
    dth1,           # dot(th1) = dth1
    dth2,           # dot(th2) = dth2
    acc_vec[0],     # dot(dth1) = acc1
    acc_vec[1]      # dot(dth2) = acc2
])

dyn_discrete = sy.Matrix(x_sym) + dt * dyn_cont # x_{t+1}

# Gradients
 
dfx_sym = dyn_discrete.jacobian(x_sym).T 
dfu_sym = dyn_discrete.jacobian(u_sym).T

# Lambdify for numerical evaluation
# syntax:  function = sy.lambdify(Input var, formula to be converted, backend library)
f_step_func = sy.lambdify([x_sym, u_sym], dyn_discrete, 'numpy')
dfx_func = sy.lambdify([x_sym, u_sym], dfx_sym, 'numpy')
dfu_func = sy.lambdify([x_sym, u_sym], dfu_sym, 'numpy')


def dynamics_euler(xx,uu):
    r"""
    Nonlinear dynamics of a acrobot model (double pendulum)

    Args
      - xx \in \R^4 state at time t
      - uu \in \R^1 input at time t
    

    Return 
      - next state xx_{t+1}
      - gradient of f wrt x, at xx,uu
      - gradient of f wrt u, at xx,uu
  
  """
    
    # Next state vector
    xxp = np.array(f_step_func(xx, uu)).squeeze() 

    # Gradients
    dfx = np.array(dfx_func(xx, uu))
    dfu = np.array(dfu_func(xx, uu))

    return xxp, dfx, dfu

#------------- Runge-Kutta 4th order method ---------------

#k_1 = dyn_cont

#x_step_2 = sy.Matrix(x_sym) + (dt / 2) * k_1 #the new point at second step
#subs_k2 = list(zip(x_sym, x_step_2)) # creating a list of pairs 
#k_2 = dyn_cont.subs(subs_k2) # computing the dynamics substituting the new state point

#x_step_3 = sy.Matrix(x_sym) + (dt / 2) * k_2 # the new point at third step
#subs_k3 = list(zip(x_sym, x_step_3)) # creating a new list of pairs
#k_3 = dyn_cont.subs(subs_k3)  # computing the dynamics substituting the new state point

#x_step_4 = sy.Matrix(x_sym) + dt * k_3 # the new point at fourth step
#subs_k4 = list(zip(x_sym, x_step_4)) # creating a new list of pairs
#k_4 = dyn_cont.subs(subs_k4)  # computing the dynamics substituting the new state point

# Discrete dynamics
#dyn_discrete_rk = sy.Matrix(x_sym) + (dt/6)*(k_1+2*k_2+2*k_3+k_4)

# Gradients
#dfx_sym_rk = dyn_discrete_rk.jacobian(x_sym).T 
#dfu_sym_rk = dyn_discrete_rk.jacobian(u_sym).T
dfx_sym_cont = dyn_cont.jacobian(x_sym).T  
dfu_sym_cont = dyn_cont.jacobian(u_sym).T

# Numerical evaluation
#f_step_func_rk = sy.lambdify([x_sym, u_sym], dyn_discrete_rk, 'numpy')
f_cont_func = sy.lambdify([x_sym, u_sym], dyn_cont, 'numpy')
#dfx_func_rk = sy.lambdify([x_sym, u_sym], dfx_sym_rk, 'numpy')
#dfu_func_rk = sy.lambdify([x_sym, u_sym], dfu_sym_rk, 'numpy')
dfx_func_cont = sy.lambdify([x_sym, u_sym], dfx_sym_cont, 'numpy')
dfu_func_cont = sy.lambdify([x_sym, u_sym], dfu_sym_cont, 'numpy')

def dynamics_rungekutta(xx,uu):
    r"""
    Nonlinear dynamics of a acrobot model (double pendulum)

    Args
      - xx \in \R^4 state at time t
      - uu \in \R^1 input at time t
    

    Return 
      - next state xx_{t+1}
      - gradient of f wrt x, at xx,uu
      - gradient of f wrt u, at xx,uu
  
  """
    
    xx = np.array(xx).reshape(-1)
    uu = np.array(uu).reshape(-1)
    
    # --- A. RK4 Numeric Iteration ---
    # Step 1: k1 = f(x, u)
    k1 = np.array(f_cont_func(xx, uu)).squeeze()
    
    # Step 2: k2 = f(x + 0.5*dt*k1, u)
    x_k2 = xx + 0.5 * dt * k1
    k2 = np.array(f_cont_func(x_k2, uu)).squeeze()
    
    # Step 3: k3 = f(x + 0.5*dt*k2, u)
    x_k3 = xx + 0.5 * dt * k2
    k3 = np.array(f_cont_func(x_k3, uu)).squeeze()
    
    # Step 4: k4 = f(x + dt*k3, u)
    x_k4 = xx + dt * k3
    k4 = np.array(f_cont_func(x_k4, uu)).squeeze()
    
    # Update of the state
    xxp_rk = xx + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    #xxp_rk = np.array(f_step_func_rk(xx, uu)).squeeze() # next state vector
    #dfx_rk = np.array(dfx_func_rk(xx, uu))
    #dfu_rk = np.array(dfu_func_rk(xx, uu))
    A_cont_T = np.array(dfx_func_cont(xx, uu)).T 
    B_cont_T = np.array(dfu_func_cont(xx, uu)).T



    dfx_rk = np.eye(ns) + dt * A_cont_T
    dfu_rk = dt * B_cont_T

    return xxp_rk, dfx_rk, dfu_rk






