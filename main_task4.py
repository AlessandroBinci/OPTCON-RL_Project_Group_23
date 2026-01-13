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
output_folder = os.path.join(current_dir, 'images', 'Task4')
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
from solver_MPC import solver_linear_MPC

# Defining final time in seconds
tf = ref_traj.tf

# Defining the discretization step value
dt = dyn.dt

# Defining the time-horizon value for MPC (we use a bigger time horizon value, i.e. four times the trajectory duration)
T_horizon  = int(tf/dt) 

# Defining the time-window size
T_window = 30

# Defining the time value to avoid index_errors
T_limit = T_horizon + T_window

# Defining number of states and inputs
ns = dyn.ns
ni = dyn.ni

# Defining the optimal trajectory variables
xx_opt_traj = np.load('optimal_traj_xx.npy')
uu_opt_traj = np.load('optimal_traj_uu.npy')

# Padding the optimal trajectory
T_padding = T_window + 5  # Un po' di margine extra
xx_last = xx_opt_traj[:, -1].reshape(-1, 1)
uu_last = uu_opt_traj[:, -1].reshape(-1, 1)

# Creiamo il blocco di padding ripetendo l'ultima colonna
xx_pad = np.repeat(xx_last, T_padding, axis=1)
uu_pad = np.repeat(uu_last, T_padding, axis=1)

# Incolliamo
xx_opt_traj_pad = np.hstack((xx_opt_traj, xx_pad))
uu_opt_traj_pad = np.hstack((uu_opt_traj, uu_pad))

# Linearize dynamics along the trajectory
AA_t = np.zeros((ns, ns, T_limit))
BB_t = np.zeros((ns, ni, T_limit))


# Finding the new A and B matrices dependent from the optimal trajectories xx and uu 
for tt in range(T_limit):
    x_t = xx_opt_traj[:, tt]
    u_t = uu_opt_traj[:, tt]
    A = (dyn.dynamics_euler(x_t, u_t)[1]).T
    B = (dyn.dynamics_euler(x_t, u_t)[2]).T
    AA_t[:, :, tt] = A
    BB_t[:, :, tt] = B

# Define LQ cost matrices
QQ = np.diag([100, 100, 1, 1])
RR = np.diag([1])
QQ_final = np.diag([100, 100, 1, 1])

# Defining the input perturbation 
initial_perturbation = 0.25

xx0_pert = xx_opt_traj[:,0].copy() + initial_perturbation

xx_MPC_track = np.zeros((ns, T_horizon)) # defining x_t
uu_MPC_track = np.zeros((ni, T_horizon)) # defining u_t
xx_MPC_track[:, 0] = xx0_pert            # defning the initial condition as perturbed
xx_mpc = np.zeros((ns, T_window, T_horizon))

for tt in range(T_horizon-1):

    # System evolution - real with MPC
    xx_mpc_actual = xx_MPC_track[:,tt] # get initial condition

    # Solve MPC problem - get first input
    uu_MPC_track[:,tt] = solver_linear_MPC(AA_t, BB_t, QQ, RR, QQ_final, xx_mpc_actual, T_horizon)[0]
    # Solve MPC problem - apply first input and get x_t+1
    xx_MPC_track[:,tt+1] = dyn.dynamics_euler(xx_MPC_track[:,tt], uu_MPC_track[:,tt])[0]

