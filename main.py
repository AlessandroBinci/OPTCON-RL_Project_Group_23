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
import matplotlib.cm as cm
import sympy as sp
import math
from mpl_toolkits import mplot3d # used for 3D plot
import cvxpy as cvx
import signal
from scipy.optimize import minimize
from matplotlib.animation import FuncAnimation
import control 
import random

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

# Variables required for plots
xx_history = []
uu_history = []

#Newton's method
for kk in range(max_iters):
    AA_kk = np.zeros((ns, ns, TT))          #initialization of matrices
    BB_kk = np.zeros((ns, ni, TT))
    qq_kk = np.zeros((ns, TT))
    rr_kk = np.zeros((ni, TT))
    cost_current = 0
    xx_history.append(xx.copy())
    uu_history.append(uu.copy())
    
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

xx_history.append(xx.copy())
uu_history.append(uu.copy())

# --- PLOT 1: Optimal Trajectory vs Desired Curve ---
# Requisito Assignment: "Optimal trajectory and desired curve"

# 1. Definizione dell'asse temporale
# Assumiamo che tf e TT siano definiti nel tuo main
time_axis = np.linspace(0, tf, TT)

# 2. Creazione della figura
plt.figure(figsize=(10, 8))

# --- Subplot 1: Primo Stato (Theta 1) ---
plt.subplot(3, 1, 1)
# Desired (Riferimento) - Linea tratteggiata nera
plt.plot(time_axis, xx_ref[0, :], 'k--', linewidth=2, label=r'$\theta_{1,des}$ (Desired)')
# Optimal (Ottima) - Linea solida colorata
plt.plot(time_axis, xx[0, :], 'b-', linewidth=2, label=r'$\theta_{1,opt}$ (Optimal)')
plt.ylabel(r'$\theta_1$ [rad]')
plt.title('Optimal Trajectory vs Desired Curve')
plt.grid(True)
plt.legend(loc='best')

# --- Subplot 2: Secondo Stato (Theta 2) ---
plt.subplot(3, 1, 2)
# Desired
plt.plot(time_axis, xx_ref[1, :], 'k--', linewidth=2, label=r'$\theta_{2,des}$ (Desired)')
# Optimal
plt.plot(time_axis, xx[1, :], 'r-', linewidth=2, label=r'$\theta_{2,opt}$ (Optimal)')
plt.ylabel(r'$\theta_2$ [rad]')
plt.grid(True)
plt.legend(loc='best')

# --- Subplot 3: Input (Coppia) ---
plt.subplot(3, 1, 3)
# Desired
plt.plot(time_axis, uu_ref[0, :], 'k--', linewidth=2, label=r'$u_{des}$ (Desired)')
# Optimal
plt.plot(time_axis, uu[0, :], 'g-', linewidth=2, label=r'$u_{opt}$ (Optimal)')
plt.ylabel(r'Torque [Nm]')
plt.xlabel(r'Time [s]')
plt.grid(True)
plt.legend(loc='best')

# Ottimizzazione spazi e salvataggio
plt.tight_layout()

# Salva l'immagine per il report (opzionale, rimuovi se non vuoi salvare)
plt.savefig('Task1_Optimal_vs_Desired.png', dpi=300)

# Mostra a video
plt.show()

# --- PLOT 1-bis: Optimal Velocities vs Desired (Zero) ---
# Visualizzazione degli stati x3 (Velocità Theta 1) e x4 (Velocità Theta 2)

plt.figure(figsize=(10, 8))

# --- Subplot 1: Terzo Stato (x3 - Velocità Theta 1) ---
plt.subplot(2, 1, 1)
# Desired (Riferimento) - Solitamente zero
plt.plot(time_axis, xx_ref[2, :], 'k--', linewidth=2, label=r'$\dot{\theta}_{1,des}$ (Desired)')
# Optimal (Ottima)
plt.plot(time_axis, xx[2, :], 'b-', linewidth=2, label=r'$\dot{\theta}_{1,opt}$ (Optimal)')
plt.ylabel(r'$\dot{\theta}_1$ [rad/s]')
plt.title('Optimal Velocities vs Desired')
plt.grid(True)
plt.legend(loc='best')

# --- Subplot 2: Quarto Stato (x4 - Velocità Theta 2) ---
plt.subplot(2, 1, 2)
# Desired (Riferimento) - Solitamente zero
plt.plot(time_axis, xx_ref[3, :], 'k--', linewidth=2, label=r'$\dot{\theta}_{2,des}$ (Desired)')
# Optimal (Ottima)
plt.plot(time_axis, xx[3, :], 'r-', linewidth=2, label=r'$\dot{\theta}_{2,opt}$ (Optimal)')
plt.ylabel(r'$\dot{\theta}_2$ [rad/s]')
plt.xlabel(r'Time [s]')
plt.grid(True)
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('Task1_Velocities.png', dpi=300) # Salva per il report
plt.show()


# 1. Define time axis
time_axis = np.linspace(0, tf, TT)

# 2. Select specific iterations to plot: [0, 17, 34, Final]
total_iters = len(xx_history)
desired_indices = [0, 1, 2, total_iters - 1]

# Filter indices to ensure they exist and avoid duplicates
indices_to_plot = []
for idx in desired_indices:
    if idx < total_iters:
        if idx not in indices_to_plot:
            indices_to_plot.append(idx)

print(f"Plotting iterations: {indices_to_plot}")

# 3. Retrieve trajectory data
xx_plot_list = [xx_history[i] for i in indices_to_plot]
uu_plot_list = [uu_history[i] for i in indices_to_plot]
num_plots = len(indices_to_plot)

# --- 4. MANUAL COLOR MANAGEMENT ---
# Qui definisci manualmente il colore per ogni curva che verrà plottata.
# L'ordine corrisponde a: [Iter 0, Iter 1, Iter 2, Finale]
# Puoi usare nomi ('red', 'blue', 'grey') o Hex Code ('#FFA500')

custom_colors = [
    'tab:orange',    # Iter 0 (Initial Guess) - Grigio per non distrarre
    'tab:blue',  # Iter 1 - Viola per contrasto
    'tab:green',  # Iter 2 - Arancione per transizione
    'tab:red'     # Final (Optimal) - Blu (Coerente col Plot 1)
]

# Sicurezza: se per caso hai meno iterazioni, tronchiamo la lista colori
if len(custom_colors) > num_plots:
    current_colors = custom_colors[:num_plots]
else:
    # Se ne mancano, ripetiamo l'ultimo (caso raro)
    current_colors = custom_colors + [custom_colors[-1]]*(num_plots-len(custom_colors))

# Se vuoi l'ottima finale in Rosso invece che Blu, cambia l'ultimo elemento della lista sopra.

LW = 1.5 # Line Width

# --- FIGURE 1: Positions and Input ---
plt.figure(figsize=(12, 10))

# Subplot 1: Theta 1 Evolution
plt.subplot(3, 1, 1)
# Reference (Nero tratteggiato)
plt.plot(time_axis, xx_ref[0, :], 'k--', linewidth=LW, label=r'$\theta_{1,des}$ (Reference)')

for k, idx in enumerate(indices_to_plot):
    # Label definition
    if idx == 0:
        lbl = 'Initial Guess (Iter 0)'
    elif idx == total_iters - 1:
        lbl = 'Optimal (Final)'
    else:
        lbl = f'Iter {idx}'
    
    # Plot using manual colors
    plt.plot(time_axis, xx_plot_list[k][0, :], 
             color=current_colors[k], linestyle='-', linewidth=LW, label=lbl)

plt.ylabel(r'$\theta_1$ [rad]')
plt.title('Evolution of Trajectories: Positions and Input')
plt.grid(True)
plt.legend(loc='best', fontsize='small')

# Subplot 2: Theta 2 Evolution
plt.subplot(3, 1, 2)
plt.plot(time_axis, xx_ref[1, :], 'k--', linewidth=LW, label=r'$\theta_{2,des}$')
for k, idx in enumerate(indices_to_plot):
    plt.plot(time_axis, xx_plot_list[k][1, :], 
             color=current_colors[k], linestyle='-', linewidth=LW)
plt.ylabel(r'$\theta_2$ [rad]')
plt.grid(True)

# Subplot 3: Input Evolution
plt.subplot(3, 1, 3)
plt.plot(time_axis, uu_ref[0, :], 'k--', linewidth=LW, label=r'$u_{des}$')
for k, idx in enumerate(indices_to_plot):
    plt.plot(time_axis, uu_plot_list[k][0, :], 
             color=current_colors[k], linestyle='-', linewidth=LW)
plt.ylabel('Torque [Nm]')
plt.xlabel('Time [s]')
plt.grid(True)

plt.tight_layout()
plt.savefig('Task1_Evolution_Pos_Input_ManualColors.png', dpi=300)
plt.show()


# --- FIGURE 2: Velocities Evolution ---
plt.figure(figsize=(10, 8))

# Subplot 1: Velocity Theta 1
plt.subplot(2, 1, 1)
plt.plot(time_axis, xx_ref[2, :], 'k--', linewidth=LW, label=r'$\dot{\theta}_{1,des}$')

for k, idx in enumerate(indices_to_plot):
    # Legend only on top
    if idx == 0:
        lbl = 'Initial Guess (Iter 0)'
    elif idx == total_iters - 1:
        lbl = 'Optimal (Final)'
    else:
        lbl = f'Iter {idx}'
        
    plt.plot(time_axis, xx_plot_list[k][2, :], 
             color=current_colors[k], linestyle='-', linewidth=LW, label=lbl)

plt.ylabel(r'$\dot{\theta}_1$ [rad/s]')
plt.title('Evolution of Trajectories: Velocities')
plt.grid(True)
plt.legend(loc='best', fontsize='small')

# Subplot 2: Velocity Theta 2
plt.subplot(2, 1, 2)
plt.plot(time_axis, xx_ref[3, :], 'k--', linewidth=LW, label=r'$\dot{\theta}_{2,des}$')
for k, idx in enumerate(indices_to_plot):
    plt.plot(time_axis, xx_plot_list[k][3, :], 
             color=current_colors[k], linestyle='-', linewidth=LW)

plt.ylabel(r'$\dot{\theta}_2$ [rad/s]')
plt.xlabel('Time [s]')
plt.grid(True)

plt.tight_layout()
plt.savefig('Task1_Evolution_Velocities_ManualColors.png', dpi=300)
plt.show()