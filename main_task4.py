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
import dynamics as dyn
import reference_curve as ref_traj
from solver_MPC import solver_linear_MPC

# Defining final time in seconds
tf = ref_traj.tf

# Defining the discretization step value
dt = dyn.dt

# Defining the time-horizon value
T_horizon  = int(tf/dt)

# Defining the time-window size
T_window = 25

# Defining the time value to avoid index_errors (+10 to be robust)
T_limit = T_horizon + T_window

# Defining the input torque limits
uu_max = 15
uu_min = -1.5

# Defining number of states and inputs
ns = dyn.ns
ni = dyn.ni

# Loading from task 2 the optimal trajectory variables
xx_opt_traj = np.load('optimal_traj_xx.npy')
uu_opt_traj = np.load('optimal_traj_uu.npy')

# Padding the optimal trajectory
xx_last = xx_opt_traj[:, -1].reshape(-1, 1) 
uu_last = uu_opt_traj[:, -1].reshape(-1, 1)

# Creating trhe padding block to add to the optimal trajectory repeating last column
xx_pad = np.repeat(xx_last, T_window, axis=1)
uu_pad = np.repeat(uu_last, T_window, axis=1)

# Attaching the padded part
xx_opt_traj_pad = np.hstack((xx_opt_traj, xx_pad))
uu_opt_traj_pad = np.hstack((uu_opt_traj, uu_pad))

# Initializing linearization dynamics matrices
AA_t = np.zeros((ns, ns, T_limit))
BB_t = np.zeros((ns, ni, T_limit))


# Finding the new A and B matrices dependent from the optimal trajectories xx and uu 
for tt in range(T_limit):
    
    # Computing "xt" and "ut" until the padded trajectory values
    x_t = xx_opt_traj_pad[:, tt]
    u_t = uu_opt_traj_pad[:, tt]
    
    # Computing A and B matrices
    A = (dyn.dynamics_euler(x_t, u_t)[1]).T
    B = (dyn.dynamics_euler(x_t, u_t)[2]).T
    
    # Assigning the A and B matrices to the corresponding time instants values
    AA_t[:, :, tt] = A
    BB_t[:, :, tt] = B

# Defining LQ cost matrices
QQ = np.diag([200, 200, 1, 1])
RR = np.diag([0.1])
QQ_final = np.diag([200, 200, 1, 1])

# Defining the input perturbation, as random, for each state
initial_perturbation = np.random.uniform(0, 0.5, 4) # Syntax: uniform(start, end, num_elem)
print(f"The initial perturbation on each state is:\n{initial_perturbation}")
xx0_pert = xx_opt_traj[:,0].copy() + initial_perturbation

# Initializing tracking trajectories
xx_MPC_track = np.zeros((ns, T_horizon)) # defining x_t
uu_MPC_track = np.zeros((ni, T_horizon)) # defining u_t

# Setting initial condition as perturbed
xx_MPC_track[:, 0] = xx0_pert

print(f"Starting MPC Simulation with a time window step N={T_window}...")

# MPC loop
for tt in range(T_horizon-1):

    # Measuring the current state
    x_mpc_curr = xx_MPC_track[:, tt]
    
    # Computing current error (delta_x)
    # The solver needs to know how far we are from the reference
    delta_x0 = x_mpc_curr - xx_opt_traj[:, tt]
    
    # Data Slicing (Preparation for the Solver)
    # Extracting the window from 'tt' to 'tt + T_window'
    AA_window = AA_t[:, :, tt : tt + T_window]
    BB_window = BB_t[:, :, tt : tt + T_window]
    uu_ref_window = uu_opt_traj_pad[:, tt : tt + T_window] # Needed for input constraints
    
    # Solve MPC problem - it returns the optimal correction "delta_u" for the current step (the first one)
    delta_u = solver_linear_MPC(AA_window, BB_window, QQ, RR, QQ_final, delta_x0, T_window, uu_ref_window, uu_max, uu_min)
    
    # Applying Input
    # u_applied = uu_reference + delta_u_optimal
    u_applied = uu_opt_traj[:, tt] + delta_u
    uu_MPC_track[:, tt] = u_applied
    
    # Solving MPC problem - apply first input and get x_t+1
    xx_MPC_track[:,tt+1] = dyn.dynamics_euler(x_mpc_curr, u_applied)[0]
    
    
# Copy for plotting consistency
xx_final = xx_MPC_track.copy()
uu_final = uu_MPC_track.copy()

# ==============================================================================
# PLOTTING RESULTS
# ==============================================================================

# Define temporal axis for plotting
time_axis = np.linspace(0, tf, T_horizon)

# --- PLOT 1: Tracking Trajectory vs Optimal Reference (Positions & Input) ---
plt.figure(figsize=(10, 8))

# Subplot 1: Theta 1
plt.subplot(3, 1, 1)
plt.plot(time_axis, xx_opt_traj[0, :], 'k--', linewidth=2, label=r'$\theta_{1,ref}$ (Optimal)')
plt.plot(time_axis, xx_final[0, :], 'b-', linewidth=2, label=r'$\theta_{1,MPC}$ (MPC)')
plt.ylabel(r'$\theta_1$ [rad]')
plt.title('MPC Tracking Trajectory vs Optimal Trajectory')
plt.grid(True)
plt.legend(loc='best')

# Subplot 2: Theta 2
plt.subplot(3, 1, 2)
plt.plot(time_axis, xx_opt_traj[1, :], 'k--', linewidth=2, label=r'$\theta_{2,ref}$ (Optimal)')
plt.plot(time_axis, xx_final[1, :], 'r-', linewidth=2, label=r'$\theta_{2,MPC}$ (MPC)')
plt.ylabel(r'$\theta_2$ [rad]')
plt.grid(True)
plt.legend(loc='best')

# Subplot 3: Control Input (Torque)
plt.subplot(3, 1, 3)
plt.plot(time_axis, uu_opt_traj[0, :], 'k--', linewidth=2, label=r'$u_{ref}$ (Optimal)')
plt.plot(time_axis, uu_final[0, :], 'g-', linewidth=2, label=r'$u_{MPC}$ (MPC)')
# Plot constraints lines (visual aid)
plt.axhline(y=uu_max, color='r', linestyle=':', label='Max Constraint')
plt.axhline(y=uu_min, color='r', linestyle=':', label='Min Constraint')
plt.ylabel(r'Torque [Nm]')
plt.xlabel(r'Time [s]')
plt.grid(True)
plt.legend(loc='best')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'Task4_Tracking_vs_Reference.png'), dpi=300)
plt.show()

# --- PLOT 1-bis: Tracking Trajectory vs Optimal Reference (Velocities) ---
plt.figure(figsize=(10, 6))

# Subplot 1: dTheta 1
plt.subplot(2, 1, 1)
plt.plot(time_axis, xx_opt_traj[2, :], 'k--', linewidth=2, label=r'$\dot{\theta}_{1,ref}$')
plt.plot(time_axis, xx_final[2, :], 'b-', linewidth=2, label=r'$\dot{\theta}_{1,MPC}$')
plt.ylabel(r'$\dot{\theta}_1$ [rad/s]')
plt.title('MPC Velocities Tracking vs Optimal Trajectory')
plt.grid(True)
plt.legend()

# Subplot 2: dTheta 2
plt.subplot(2, 1, 2)
plt.plot(time_axis, xx_opt_traj[3, :], 'k--', linewidth=2, label=r'$\dot{\theta}_{2,ref}$')
plt.plot(time_axis, xx_final[3, :], 'r-', linewidth=2, label=r'$\dot{\theta}_{2,MPC}$')
plt.ylabel(r'$\dot{\theta}_2$ [rad/s]')
plt.xlabel('Time [s]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'Task4_Velocities.png'), dpi=300)
plt.show()

# --- PLOT 2: Tracking Error for each state and for the input ---
# Visualization of the tracking errors

# Plot tracking errors for state variables
state_labels = [r"${\theta}_1$", r"${\theta}_2$", r"$\dot{\theta}_1$", r"$\dot{\theta}_2$"]

for i, label in enumerate(state_labels):
    plt.figure(figsize=(8, 5))
    plt.plot(time_axis, np.abs(xx_final[i, :] - xx_opt_traj[i, :]), label=f"Tracking Error in {label}")
    plt.title(f"Tracking Error in {label}")
    plt.xlabel("Time [s]")
    plt.ylabel(f"Error in {label}")
    plt.grid()
    plt.legend()
    plot_name = f"task4_{label.replace('$', '').replace('\\', '').replace('.', '').replace('{', '').replace('}', '').replace('_', '').lower()}_tracking_error.png"
    plt.savefig(os.path.join(output_folder,plot_name), dpi=300) # Save for the report
    print(plot_name)
    plt.show()

# Plot tracking error for input variable
input_tracking_error = np.abs(uu_final - uu_opt_traj)
plt.figure(figsize=(8, 5))
plt.plot(time_axis, input_tracking_error[0, :], label="Tracking Error in Input (u)")
plt.title("Tracking Error in Input (u)")
plt.xlabel("Time [s]")
plt.ylabel("Error in u")
plt.grid()
plt.legend()
plt.savefig(os.path.join(output_folder,'Task4_Input_TrackError.png'), dpi=300) # Save for the report

plt.show()