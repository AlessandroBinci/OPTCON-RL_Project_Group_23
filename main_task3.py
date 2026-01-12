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
output_folder = os.path.join(current_dir, 'images', 'Task3')
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

# Defining final time in seconds
tf = ref_traj.tf

# Defining the discretization step value
dt = dyn.dt

# Defining the time-horizon value
T_horizon  = int(tf/dt)

# Defining number of states and inputs
ns = dyn.ns
ni = dyn.ni

# Defining the optimal trajectory variables

xx_opt_traj = np.load('optimal_traj_xx.npy')
uu_opt_traj = np.load('optimal_traj_uu.npy')


# Linearize dynamics along the trajectory
AA_t = np.zeros((ns, ns, T_horizon))
BBinch_t = np.zeros((ns, ni, T_horizon))

# Finding the new A and B matrices dependent from the optimal trajectories xx and uu 
for tt in range(T_horizon):
    x_t = xx_opt_traj[:, tt]
    u_t = uu_opt_traj[:, tt]
    A = (dyn.dynamics_euler(x_t, u_t)[1]).T
    B = (dyn.dynamics_euler(x_t, u_t)[2]).T
    AA_t[:, :, tt] = A
    BBinch_t[:, :, tt] = B

# Define LQR cost matrices
QQreg = np.diag([100, 100, 1, 1])
RRreg = np.diag([1])
QQreg_final = np.diag([100, 100, 1, 1])
SSreg = np.zeros((ni, ns))

qq_reg = np.zeros((ns, T_horizon))
rr_reg = np.zeros((ni, T_horizon))
qq_Terminal = np.zeros(ns)

# Solve the LQR problem
KK = ltv_LQR(AA_t, BBinch_t, QQreg, RRreg, SSreg, QQreg_final, T_horizon, np.zeros((ns)), qq_reg, rr_reg, qq_Terminal)[0]

# Defining the input perturbation 
initial_perturbation = 0.25

xx0_pert = xx_opt_traj[:,0].copy() + initial_perturbation

xx_LQR_track = np.zeros((ns, T_horizon)) # defining x_t
uu_LQR_track = np.zeros((ni, T_horizon)) # defining u_t
xx_LQR_track[:, 0] = xx0_pert            # defning the initial condition as perturbed
state_error = np.zeros((ns, T_horizon))

for tt in range(T_horizon-1):

    state_error[:,tt] = xx_LQR_track[:,tt] - xx_opt_traj[:,tt]

    uu_LQR_track[:,tt] = uu_opt_traj[:,tt] + KK[:,:,tt] @ state_error[:,tt]

    xx_LQR_track[:,tt+1] = dyn.dynamics_euler(xx_LQR_track[:, tt], uu_LQR_track[:, tt])[0]


xx_final = xx_LQR_track.copy()
uu_final = uu_LQR_track.copy()


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
plt.plot(time_axis, xx_final[0, :], 'b-', linewidth=2, label=r'$\theta_{1,track}$ (Tracking)')
plt.ylabel(r'$\theta_1$ [rad]')
plt.title('Tracking trajectory vs Optimal Trajectory')
plt.grid(True)
plt.legend(loc='best')

# --- Subplot 2: Second State (Theta 2) ---
plt.subplot(3, 1, 2)
# Optimal traj.
plt.plot(time_axis, xx_opt_traj[1, :], 'k--', linewidth=2, label=r'$\theta_{2,opt}$ (Optimal)')
# Tracking traj.
plt.plot(time_axis, xx_final[1, :], 'r-', linewidth=2, label=r'$\theta_{2,track}$ (Tracking)')
plt.ylabel(r'$\theta_2$ [rad]')
plt.grid(True)
plt.legend(loc='best')

# --- Subplot 3: Input ---
plt.subplot(3, 1, 3)
# Optimal traj.
plt.plot(time_axis, uu_opt_traj[0, :], 'k--', linewidth=2, label=r'$u_{opt}$ (Optimal)')
# Tracking traj.
plt.plot(time_axis, uu_opt_traj[0, :], 'g-', linewidth=2, label=r'$u_{track}$ (Tracking)')
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
plt.plot(time_axis, xx_final[2, :], 'b-', linewidth=2, label=r'$\dot{\theta}_{1,track}$ (Tracking)')
plt.ylabel(r'$\dot{\theta}_1$ [rad/s]')
plt.title('Tracking velocities vs Optimal Velocities')
plt.grid(True)
plt.legend(loc='best')

# --- Subplot 2: Fourth State (x4 - Velocity Theta 2) ---
plt.subplot(2, 1, 2)
# Optimal traj. (generally zero)
plt.plot(time_axis, xx_opt_traj[3, :], 'k--', linewidth=2, label=r'$\dot{\theta}_{2,opt}$ (Optimal)')
# Tracking traj.
plt.plot(time_axis, xx_final[3, :], 'r-', linewidth=2, label=r'$\dot{\theta}_{2,track}$ (Tracking)')
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
    plt.plot(time_axis, np.abs(xx_final[i, :] - xx_opt_traj[i, :]), label=f"Tracking Error in {label}")
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
input_tracking_error = np.abs(uu_final - uu_opt_traj)
plt.figure(figsize=(8, 5))
plt.plot(time_axis, input_tracking_error[0, :], label="Tracking Error in Input (u)")
plt.title("Tracking Error in Input (u)")
plt.xlabel("time [s]")
plt.ylabel("Error in u")
plt.grid()
plt.legend()
plt.savefig(os.path.join(output_folder,'Task3_Input_TrackError.png'), dpi=300) # Save for the report

plt.show()