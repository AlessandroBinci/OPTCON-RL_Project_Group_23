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
output_folder = os.path.join(current_dir, 'images', 'Task2')
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

from main_task2 import xx, uu

# Defining final time in seconds
tf = ref_traj.tf

# Defining the discretization step value
dt = dyn.dt

# Defining the time-horizon value
T_horizon  = int(tf/dt)

# Defining number of states and inputs
ns = dyn.ns
ni = dyn.ni

# Linearize dynamics along the trajectory
AA_t = np.zeros((ns, ns, T_horizon))
BBinch_t = np.zeros((ns, ni, T_horizon))

#DA FINIRE -> STEP 1 DI 3 PER LA TASK 3 -> CALCOLARE A E B NUOVE
for tt in range(T_horizon):
    x_t = xx[:, tt]
    u_t = uu[0, tt]
    _, A, B = dyn.dynamics_euler(x_t, u_t)
    AA_t[:, :, tt] = A
    BBinch_t[:, :, tt] = B



