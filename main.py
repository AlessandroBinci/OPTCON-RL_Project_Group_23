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

# Defining final time in seconds
tf = ref_traj.tf

# Defining the discretization step value
dt = dyn.dt

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