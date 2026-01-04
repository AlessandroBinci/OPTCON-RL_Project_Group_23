# Optimal Control and Reinforcement Learning Project 2025-2026
# Group 23: Alessandro Binci, Alessandro Tampieri, Lorenzo Tucci
# Problem 2 with set ofparameters 1

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

xx = xx_ref[:,0].copy() #useful to ceate a copy of the target 
uu = uu_ref.copy()

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

    for tt in range(TT):
        AA_kk[:,:, tt] = dyn.dynamics_euler(xx[:, tt], uu[:, tt]) [1]
        BB_kk[:,:, tt] = dyn.dynamics_euler(xx[:, tt], uu[:, tt]) [2]
        cost_actual = cst.stage_cost(xx[:,tt],uu[:,tt],xx_ref[:,tt],uu_ref[:,tt]) [0]
        qq_kk[:,TT] = cst.stage_cost(xx[:,tt],uu[:,tt],xx_ref[:,tt],uu_ref[:,tt]) [1]
        rr_kk[:,TT] = cst.stage_cost(xx[:,tt],uu[:,tt],xx_ref[:,tt],uu_ref[:,tt]) [2]
        cost_current += cost_actual

    termcost_actual = cst.termcost(xx[:,-1], xx_ref[:,-1]) [0]   
    qqT_kk = cst.termcost(xx[:,-1], xx_ref[:,-1]) [1]
    cost_current += termcost_actual
    KK, sigma = ltv_LQR(AA_kk, BB_kk, QQ, RR, SS, QQ, TT, np.zeros((ns)), qq_kk, rr_kk, qqT_kk) [0:2]









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