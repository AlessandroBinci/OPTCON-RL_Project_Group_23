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
# Add 'docs/Task2' folder path to store the images
output_folder = os.path.join(current_dir, 'images', 'Task4bis')
# If the folder does not exists, create one
if not os.path.exists(output_folder):
    print(f"The folder didn't exist, creating folder here:{output_folder}")
    os.makedirs(output_folder)
else:
    print(f"Store images in: {output_folder}")


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
from solver4 import solver_lin_mpc


# Defining final time in seconds
tf = ref_traj.tf

# Defining the discretization step value
dt = dyn.dt

# Defining the time-horizon value for MPC (we use a bigger time horizon value, i.e. four times the trajectory duration)
T_horizon  = int(tf/dt) 

# Defining the time-window size
T_window = 25

# Defining the time value to avoid index_errors (+10 to be robust)
T_limit = T_horizon + T_window + 10

# Defining number of states and inputs
ns = dyn.ns
ni = dyn.ni

# Defining the optimal trajectory variables
xx_opt_traj = np.load('optimal_traj_xx.npy')
uu_opt_traj = np.load('optimal_traj_uu.npy')

# Creiamo array estesi vuoti
xx_ref_pad = np.zeros((ns, T_limit))
uu_ref_pad = np.zeros((ni, T_limit))

# Copying original data
len_data = xx_opt_traj.shape[1]
xx_ref_pad[:, :len_data] = xx_opt_traj
uu_ref_pad[:, :len_data] = uu_opt_traj

# Riempiamo la parte finale ripetendo l'ULTIMO valore (Equilibrio)
for k in range(len_data, T_limit):
    xx_ref_pad[:, k] = xx_opt_traj[:, -1]
    uu_ref_pad[:, k] = uu_opt_traj[:, -1]

# Linearize dynamics along the trajectory
AA_t = np.zeros((ns, ns, T_limit))
BB_t = np.zeros((ns, ni, T_limit))


# Finding the new A and B matrices dependent from the optimal trajectories xx and uu 
for tt in range(T_limit):
    x_t = xx_ref_pad[:, tt]
    u_t = uu_ref_pad[:, tt]
    AA_t[:, :, tt] = (dyn.dynamics_euler(x_t, u_t)[1]).T
    BB_t[:, :, tt] = (dyn.dynamics_euler(x_t, u_t)[2]).T

# Define LQ cost matrices
QQ = np.diag([200, 200, 1, 1])
RR = np.diag([0.1])
QQ_final = np.diag([200, 200, 1, 1])

# Defining the input perturbation 
initial_perturbation = 0.25

xx0_pert = xx_opt_traj[:,0].copy() + initial_perturbation

xx_MPC_track = np.zeros((ns, T_horizon)) # defining x_t
uu_MPC_track = np.zeros((ni, T_horizon)) # defining u_t
xx_MPC_track[:, 0] = xx0_pert            # defning the initial condition as perturbed

AA_actual = np.zeros((ns, ns, T_window))
BB_actual = np.zeros((ns, ni, T_window))

# MPC loop 
for tt in range(T_horizon-1):

    AA_actual = AA_t[:,:,tt:tt+T_window]
    BB_actual = BB_t[:,:,tt:tt+T_window]
    
    # System evolution - real with MPC
    xx_mpc_actual = xx_MPC_track[:,tt] # get initial condition

    xx_ref_actual = xx_ref_pad[:, tt]
    delta_x = xx_mpc_actual - xx_ref_actual
    # Solve MPC problem - get first input
    delta_u = solver_lin_mpc(AA_actual, BB_actual, QQ, RR, QQ_final, delta_x, T_window)
    uu_ref_actual = uu_ref_pad[:, tt]
    uu_MPC_track[:, tt] = uu_ref_actual + delta_u
    # Solve MPC problem - apply first input and get x_t+1
    xx_MPC_track[:,tt+1] = dyn.dynamics_euler(xx_MPC_track[:,tt], uu_MPC_track[:,tt])[0]
    

# Copying for plotting purposes
xx_final_mpc = xx_MPC_track.copy()
uu_final_mpc = uu_MPC_track.copy()

# --- PLOT: Tracking Trajectory vs Optimal Trajectory ---
# Requirement: "Tracking trajectory and optimal trajectory"

# Temporal axis definition
# We assume that 'tf' and 'T_horizon' are defined 
time_axis = np.linspace(0, tf, T_horizon)

# Figure creation
plt.figure(figsize=(10, 8))

# --- Subplot 1: First State (Theta 1) ---
plt.subplot(3, 1, 1)
# Optimal traj. - Black Dashed Line
plt.plot(time_axis, xx_opt_traj[0, :], 'k--', linewidth=2, label=r'$\theta_{1,opt}$ (Optimal)')
# Tracking traj. - Solid Colored Line
plt.plot(time_axis, xx_final_mpc[0, :], 'b-', linewidth=2, label=r'$\theta_{1,track}$ (Tracking)')
plt.ylabel(r'$\theta_1$ [rad]')
plt.title('Tracking trajectory vs Optimal Trajectory')
plt.grid(True)
plt.legend(loc='best')

# --- Subplot 2: Second State (Theta 2) ---
plt.subplot(3, 1, 2)
# Optimal traj.
plt.plot(time_axis, xx_opt_traj[1, :], 'k--', linewidth=2, label=r'$\theta_{2,opt}$ (Optimal)')
# Tracking traj.
plt.plot(time_axis, xx_final_mpc[1, :], 'r-', linewidth=2, label=r'$\theta_{2,track}$ (Tracking)')
plt.ylabel(r'$\theta_2$ [rad]')
plt.grid(True)
plt.legend(loc='best')

# --- Subplot 3: Input ---
plt.subplot(3, 1, 3)
# Optimal traj.
plt.plot(time_axis, uu_opt_traj[0, :], 'k--', linewidth=2, label=r'$u_{opt}$ (Optimal)')
# Tracking traj.
plt.plot(time_axis, uu_final_mpc[0, :], 'g-', linewidth=2, label=r'$u_{track}$ (Tracking)')
plt.ylabel(r'Torque [Nm]')
plt.xlabel(r'Time [s]')
plt.grid(True)
plt.legend(loc='best')

# Optimization of the spaces and storage
plt.tight_layout()

# Save the image in the correct folder
plt.savefig(os.path.join(output_folder,'Task3_Tracking_vs_Optimal.png'), dpi=300)
# Print on screen
plt.show()

# --- PLOT 1-bis: Tracking Velocities vs Optimal Velocities ---
# Visualization of the states x3 (Velocity Theta 1) and x4 (Velocity Theta 2)

plt.figure(figsize=(10, 8))

# --- Subplot 1: Third State (x3 - Velocity Theta 1) ---
plt.subplot(2, 1, 1)
# Optimal traj. (generally zero)
plt.plot(time_axis, xx_opt_traj[2, :], 'k--', linewidth=2, label=r'$\dot{\theta}_{1,opt}$ (Optimal)')
# Tracking traj.
plt.plot(time_axis, xx_final_mpc[2, :], 'b-', linewidth=2, label=r'$\dot{\theta}_{1,track}$ (Tracking)')
plt.ylabel(r'$\dot{\theta}_1$ [rad/s]')
plt.title('Tracking velocities vs Optimal Velocities')
plt.grid(True)
plt.legend(loc='best')

# --- Subplot 2: Fourth State (x4 - Velocity Theta 2) ---
plt.subplot(2, 1, 2)
# Optimal traj. (generally zero)
plt.plot(time_axis, xx_opt_traj[3, :], 'k--', linewidth=2, label=r'$\dot{\theta}_{2,opt}$ (Optimal)')
# Tracking traj.
plt.plot(time_axis, xx_final_mpc[3, :], 'r-', linewidth=2, label=r'$\dot{\theta}_{2,track}$ (Tracking)')
plt.ylabel(r'$\dot{\theta}_2$ [rad/s]')
plt.xlabel(r'Time [s]')
plt.grid(True)
plt.legend(loc='best')

plt.tight_layout()
plt.savefig(os.path.join(output_folder,'Task3_Velocities.png'), dpi=300) # Save for the report
plt.show()



# Plot tracking error for each state

state_labels = ["θ1", "θ2", r"$\dot{\theta}_1$", r"$\dot{\theta}_2$"]

for i, label in enumerate(state_labels):
    plt.figure(figsize=(8, 5))
    plt.plot(time_axis, np.abs(xx_final_mpc[i, :] - xx_opt_traj[i, :]), label=f"Tracking Error in {label}")
    plt.title(f"Tracking Error in {label}")
    plt.xlabel("time [s]")
    plt.ylabel(f"Error in {label}")
    plt.grid()
    plt.legend()
    plot_name = f"task3_{label.replace('$', '').replace('\\', '').replace('.', '').replace('{', '').replace('}', '').replace('_', '').lower()}_tracking_error.png"
    plt.savefig(os.path.join(output_folder,plot_name), dpi=300) # Save for the report
    print(plot_name)
    plt.show()

# Plot tracking error for input
input_tracking_error = np.abs(uu_final_mpc - uu_opt_traj)
plt.figure(figsize=(8, 5))
plt.plot(time_axis, input_tracking_error[0, :], label="Tracking Error in Input (u)")
plt.title("Tracking Error in Input (u)")
plt.xlabel("time [s]")
plt.ylabel("Error in u")
plt.grid()
plt.legend()
plt.savefig(os.path.join(output_folder,'Task3_Input_TrackError.png'), dpi=300) # Save for the report

plt.show()