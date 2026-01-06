# Optimal Control and Reinforcement Learning Project 2025-2026
# Group 23: Alessandro Binci, Alessandro Tampieri, Lorenzo Tucci
# Problem 2 with set of parameters 1

import sys
import os

### --- ADDED TO USE SRC FOLDER --- ###
# In this way we obtain tha path of the current folder
current_dir = os.path.dirname(os.path.abspath(__file__))
# We build the path to 'src' folder
src_path = os.path.join(current_dir, 'src')
# Add 'src' folder tho the paths where Python search the modules
sys.path.append(src_path)

### --- IMPORTS --- ###
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math
from mpl_toolkits import mplot3d # used for 3D plot
import cvxpy as cvx
import signal
from scipy.optimize import minimize
from matplotlib.animation import FuncAnimation
import control 

import dynamics as dyn
import equilibrium as eq
import reference_curve as ref_traj
import cost as cst
from ltv_solver_LQR import ltv_LQR

# Defining final time in seconds
tf = ref_traj.tf

# Defining the discretization step value
dt = dyn.dt

# Defining the time-horizon value
TT = int(tf/dt)

# Defining number of states and inputs
ns = dyn.ns
ni = dyn.ni

# Defining the initial angular position of the first link and the desired torque for the first equilibrium point
th1_init= np.deg2rad(45)
tau1_des = 0

# Defining the final angular position of the first link for the second equilibrium point

th1_final = np.deg2rad(90)

# Computing the reference trajectory
xx_ref, uu_ref = ref_traj.gen(tf, dt, ns, ni, th1_init, tau1_des, th1_final)

# Defining the initial guess

xx = np.zeros((ns, TT))
uu = np.zeros((ni, TT))

for tt in range(TT):
    xx[:, tt] = xx_ref[:, 0]  # It always remains to x_start (Eq 1)
    uu[:, tt] = uu_ref[:, 0]  # Always applays u_start (Eq 1)

# Defining the number of max iterations
max_iters = 50

# Defining the matrices
QQ = cst.QQ
RR = cst.RR
SS = np.zeros((ni, ns))

#Newton's method
for kk in range(max_iters):
    AA_kk = np.zeros((ns, ns, TT))          #initialization of matrices
    BB_kk = np.zeros((ns, ni, TT))
    qq_kk = np.zeros((ns, TT))
    rr_kk = np.zeros((ni, TT))
    cost_current = 0
    
    for tt in range(TT-1):
        AA_kk[:,:, tt] = (dyn.dynamics_euler(xx[:, tt], uu[:, tt]) [1]).T
        BB_kk[:,:, tt] = (dyn.dynamics_euler(xx[:, tt], uu[:, tt]) [2]).T
        cost_actual = cst.stage_cost(xx[:,tt],uu[:,tt],xx_ref[:,tt],uu_ref[:,tt]) [0]
        qq_kk[:,tt] = cst.stage_cost(xx[:,tt],uu[:,tt],xx_ref[:,tt],uu_ref[:,tt]) [1]
        rr_kk[:,tt] = cst.stage_cost(xx[:,tt],uu[:,tt],xx_ref[:,tt],uu_ref[:,tt]) [2]
        cost_current += cost_actual

    termcost_actual = cst.termcost(xx[:,-1], xx_ref[:,-1],QQ) [0]   
    qqT_kk = cst.termcost(xx[:,-1], xx_ref[:,-1],QQ) [1]
    cost_current += termcost_actual
    KK, sigma = ltv_LQR(AA_kk, BB_kk, QQ, RR, SS, QQ, TT, np.zeros((ns)), qq_kk, rr_kk, qqT_kk) [0:2]
    dx_lin , du_lin = ltv_LQR(AA_kk, BB_kk, QQ, RR, SS, QQ, TT, np.zeros((ns)), qq_kk, rr_kk, qqT_kk) [3:]

    #Armijo
    gamma = 1
    beta = 0.7
    cc = 0.5
    max_armijo_iters = 10
    ii = 1
    slope = 0
    for tt in range(TT):
         # Dot Product of Gradient_x * Delta_x_lin
        slope += np.dot(qq_kk[:, tt], dx_lin[:, tt])
    
        # If we are not in the last step, we add Gradient_u * Delta_u_lin
        if tt < TT - 1:
            slope += np.dot(rr_kk[:, tt], du_lin[:, tt])

    # Add final term (Gradient_x_T * Delta_x_T)
    slope += np.dot(qqT_kk, dx_lin[:, -1])

    while ii < max_armijo_iters:

        # Temporary solution update

        xx_temp = np.zeros((ns,TT))
        uu_temp=np.zeros((ni,TT))
        xx_temp[:,0] = xx[:, 0]
        cost_temp = 0
        for tt in range(TT-1):

            # This is the term: (x_t)^k+1 - (x_t)^k
            delta_x = xx_temp[:, tt] - xx[:, tt]
        
            # u_new = u_old + gamma * sigma + K * delta_x
            # NOTE: u_old è uu[:,tt], sigma is the feedforward, K is the feedback
            ff_term = gamma * sigma[:, tt]
            fb_term = KK[:, :, tt] @ delta_x
        
            uu_temp[:, tt] = uu[:, tt] + ff_term + fb_term

            # dynamics_euler gives [x_next, A, B], we take only [0]
            xx_temp[:, tt+1] = dyn.dynamics_euler(xx_temp[:, tt], uu_temp[:, tt])[0]

            # stage_cost gives [cost, grad_x, grad_u], we take only [0]
            step_c = cst.stage_cost(xx_temp[:, tt], uu_temp[:, tt], xx_ref[:, tt], uu_ref[:, tt])[0]
            cost_temp += step_c
        
        # We compute final cost on the reached final state
        term_c = cst.termcost(xx_temp[:, -1], xx_ref[:, -1], QQ)[0]
        cost_temp += term_c
        if cost_temp >= cost_current  + cc*gamma*slope:
                  # update the stepsize
                  gamma = beta* gamma
                  ii += 1
        else:
            xx = xx_temp.copy()
            uu = uu_temp.copy()

            break


# --- PLOTTING RESULTS ---
# We create a time vector which is coherent with the dimensions
time_axis = np.linspace(0, tf, TT)

plt.figure(figsize=(12, 10))

# --- 1. First State (x1 - Theta 1) ---
plt.subplot(3, 1, 1)
plt.plot(time_axis, np.rad2deg(xx_ref[0, :]), 'k--', linewidth=2, label=r'$\theta_{1,ref}$ (Desiderata)')
plt.plot(time_axis, np.rad2deg(xx[0, :]), 'b-', linewidth=2, label=r'$\theta_{1,opt}$ (Ottima)')
plt.ylabel(r'$\theta_1$ [deg]')
plt.title('Confronto Traiettorie: Riferimento vs Ottimo')
plt.grid(True)
plt.legend(loc='best')

# --- 2. Second State (x2 - Theta 2 or Angular Velocity) ---
# NOTE: If your system has 2 states (es. pos and vel, or theta1 and theta2), we plot the second one.
if ns > 1:
    plt.subplot(3, 1, 2)
    # Se il secondo stato è una velocità, lasciamo rad/s, se è un angolo convertiamo.
    # Assumo sia un angolo o velocità, qui plotto il valore puro o convertito se necessario.
    # Per coerenza col primo plot, mostro il valore numerico diretto o convertito in gradi se è un angolo.
    plt.plot(time_axis, np.rad2deg(xx_ref[1, :]), 'k--', linewidth=2, label=r'$x_{2,ref}$')
    plt.plot(time_axis, np.rad2deg(xx[1, :]), 'r-', linewidth=2, label=r'$x_{2,opt}$')
    plt.ylabel(r'$x_2$ [deg o unit]') # Adatta l'etichetta alla tua dinamica
    plt.grid(True)
    plt.legend(loc='best')

# --- 3. Input (u - Coppia) ---
plt.subplot(3, 1, 3)
plt.plot(time_axis, uu_ref[0, :], 'k--', linewidth=2, label=r'$u_{ref}$ (Desiderata)')
plt.plot(time_axis, uu[0, :], 'g-', linewidth=2, label=r'$u_{opt}$ (Ottima)')
plt.ylabel('Input Torque [Nm]')
plt.xlabel('Time [s]')
plt.grid(True)
plt.legend(loc='best')

plt.tight_layout()
plt.show()

# --- 4. Plot Errore (Opzionale ma utile) ---
plt.figure(figsize=(10, 5))
err_norm = np.linalg.norm(xx - xx_ref, axis=0)
plt.plot(time_axis, err_norm, 'm-', linewidth=2)
plt.title('Norma dell\'errore di stato $||x - x_{ref}||$')
plt.xlabel('Time [s]')
plt.ylabel('Error Norm')
plt.grid(True)
plt.yscale('log') # Scala logaritmica utile per vedere la convergenza fine
plt.show()









# Asse temporale
time = np.linspace(0, tf, xx_ref.shape[1])

# --- 3. PLOTTING ---
plt.figure(figsize=(10, 10))

# --- PLOT THETA 1 (Stato 0) ---
plt.subplot(3, 1, 1)
# Convertiamo in gradi SOLO per il grafico
plt.plot(time, np.rad2deg(xx_ref[0, :]), 'b-', linewidth=2, label=r'$\theta_1$ Rif')
plt.ylabel(r'$\theta_1$ [deg]')
plt.title('Traiettoria di Riferimento (Step)')
plt.grid(True)
plt.legend()

# --- PLOT THETA 2 (Stato 1) ---
# Controlliamo che esista almeno un secondo stato
if ns > 1:
    plt.subplot(3, 1, 2)
    plt.plot(time, np.rad2deg(xx_ref[1, :]), 'r-', linewidth=2, label=r'$\theta_2$ Rif')
    plt.ylabel(r'$\theta_2$ [deg]')
    plt.grid(True)
    plt.legend()

# --- PLOT INPUT (Coppia) ---
plt.subplot(3, 1, 3)
# L'input è solitamente in Nm, non si converte in gradi!
plt.plot(time, uu_ref[0, :], 'g-', linewidth=2, label=r'$u_1$ (Coppia)')

# Se c'è un secondo input, lo plottiamo
if ni > 1:
    plt.plot(time, uu_ref[1, :], 'orange', linestyle='--', linewidth=2, label=r'$u_2$')

plt.ylabel('Input [Nm]')
plt.xlabel('Tempo [s]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()