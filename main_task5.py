# Optimal Control and Reinforcement Learning Project 2025-2026
# Group 23: Alessandro Binci, Alessandro Tampieri, Lorenzo Tucci
# Realistic Animation of the Flexible Link (based on main_task3.py trajectory tracking)

import sys
import os

### --- ADDED TO USE SRC FOLDER --- ###
# In this way we obtain tha path of the current folder
current_dir = os.path.dirname(os.path.abspath(__file__))
# We build the path to 'src' folder
src_path = os.path.join(current_dir, 'src')
# Add 'src' folder tho the paths where Python search the modules
sys.path.append(src_path)

# --- ADDED TO STORE ANIMATION ---
# Add 'images/Task5' folder path to store the animation
output_folder = os.path.join(current_dir, 'images', 'Task5')
# If the folder does not exists, create one
if not os.path.exists(output_folder):
    print(f"The folder didn't exist, creating folder here:{output_folder}")
    os.makedirs(output_folder)
else:
    print(f"Store images in: {output_folder}")

### --- IMPORTS --- ###
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches # To draw geometric forms (polygons and cirles)
import dynamics as dyn
from ltv_solver_LQR import ltv_LQR
from matplotlib.animation import PillowWriter # Writer to save the animation as .gif

#==============================================================================
# DATA GENERATION & SIMULATION
#==============================================================================
# Running the simulation logic from Task 3
# Generating the optimal trajectory of the perturbated system controlled by LQR

# Loading optimal trajectories computed in Task 2
try:
    xx_opt_traj = np.load('optimal_traj_xx.npy')
    uu_opt_traj = np.load('optimal_traj_uu.npy')
except FileNotFoundError:
    print("Error: Optimal trajectory files (.npy) not found. Run Task 2 first.")
    sys.exit()

# System parameters retrieval
dt = dyn.dt
ns = dyn.ns
ni = dyn.ni
T_horizon = xx_opt_traj.shape[1] # Horizon length based on loaded data

AA_t = np.zeros((ns, ns, T_horizon))
BB_t = np.zeros((ns, ni, T_horizon))

# Linearize dynamics along the optimal trajectory
# Computing the Jacobians A_t and B_t for every time step along the optimal path
for tt in range(T_horizon):
    x_t = xx_opt_traj[:, tt]
    u_t = uu_opt_traj[:, tt]
    
    # Dynamics_euler returns [x_next, A, B]. Taking [1] for A and [2] for B.
    # Transposing with (.T) because the function returns gradients, we need Jacobians.
    AA_t[:, :, tt] = (dyn.dynamics_euler(x_t, u_t)[1]).T
    BB_t[:, :, tt] = (dyn.dynamics_euler(x_t, u_t)[2]).T

# Defining LQR cost matrices (Same tuning as Task 3)
QQreg = np.diag([100, 100, 1, 1])
RRreg = np.diag([1])
QQreg_final = np.diag([100, 100, 1, 1])
SSreg = np.zeros((ni, ns))

# Compute the Time-Varying LQR Feedback Gain K_t and solving Riccati Difference Equation backward in time
KK = ltv_LQR(AA_t, BB_t, QQreg, RRreg, SSreg, QQreg_final, T_horizon, np.zeros(ns), np.zeros((ns,T_horizon)), np.zeros((ni,T_horizon)), np.zeros(ns))[0]


# --- CLOSED LOOP SIMULATION ---

# Defining the input perturbation, as random, for each state
initial_perturbation = np.random.uniform(0, 0.5, 4) # Syntax: uniform(start, end, num_elem)

# Initializing simulation arrays
xx_sim = np.zeros((ns, T_horizon)) # defining x_t
uu_sim = np.zeros((ni, T_horizon)) # defining u_t

# Apply perturbation to the initial optimal state
print(f"The initial perturbation on each state is:\n{initial_perturbation}")
xx_sim[:,0] = xx_opt_traj[:,0].copy() + initial_perturbation

# Simulate the non-linear system with LQR feedback
for tt in range(T_horizon - 1):
    error_t = xx_sim[:, tt] - xx_opt_traj[:, tt]
    uu_sim[:, tt] = uu_opt_traj[:, tt] + KK[:, :, tt] @ error_t
    xx_sim[:, tt + 1] = dyn.dynamics_euler(xx_sim[:, tt], uu_sim[:, tt])[0]


#==============================================================================
# 2. ANIMATION SETUP
#==============================================================================
# Defining visual properties and geometry for animation

# Retrieving physical lengths of the links
l1 = dyn.l_1
l2 = dyn.l_2

# --- Visual Styling Parameters (To match the CAD look) ---
link_width = 0.08       # Thickness of the rectangular bars representing the links
joint_radius = 0.06     # Radius of the circular joints
link_color = '#C0C0C0'  # Silver/Light Gray for links
joint_color = '#808080' # Darker Gray for joints
edge_color = 'black'    # Black outline for definition

# --- Helper Function for Forward Kinematics ---
def forward_kinematics(state):
    r"""
    Computes (x,y) coordinates of joints based on angles theta1, theta2.
    
    Args:
        state: [theta1, theta2, dtheta1, dtheta2]
    
    Returns:
        Arrays containing the coordinates for Base (0,0), Elbow (p1) and Tip (p2)
    """
    
    
    th1 = state[0]
    th2 = state[1]
    
    # Joint 1 (Elbow) position
    # Note: Y is negative down (-cos) assuming 0 angle is vertical down
    x1 = l1 * np.sin(th1)
    y1 = -l1 * np.cos(th1)
    
    # Joint 2 (End-Effector) position
    # The angle of the second link is relative to the first one (th1 + th2)
    x2 = x1 + l2 * np.sin(th1 + th2)
    y2 = y1 - l2 * np.cos(th1 + th2)
    
    return np.array([0,0]), np.array([x1, y1]), np.array([x2, y2])

# --- Helper Function for Geometry ---
def get_thick_link_corners(p_start, p_end, width):
    r"""
    Calculates the 4 corners of a thick rectangular bar connecting p_start and p_end.
    This is used to draw the links as Polygons instead of simple lines.
    
    Math logic:
    1. Find vector from start to end.
    2. Compute perpendicular normal vector.
    3. Offset points by +/- half width along the normal vector.
    """
    
    # Vector along the link
    vec = p_end - p_start
    length = np.linalg.norm(vec)
    
    # Unit vector along the link
    unit_vec = vec / length
    
    # Perpendicular vector (normal)
    perp_vec = np.array([-unit_vec[1], unit_vec[0]])
    
    # Half width to offset from center line
    hw = width / 2.0
    
    # The 4 corners of the rectangle centered on the line segment
    c1 = p_start + perp_vec * hw
    c2 = p_start - perp_vec * hw
    c3 = p_end - perp_vec * hw
    c4 = p_end + perp_vec * hw
    
    return np.array([c1, c2, c3, c4])

# --- Figure Setup ---
# Initializing the Matplotlib figure
fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')

# Setting plot limits based on total robot reach (l1 + l2) with some margin
limit = (l1 + l2) * 1.2
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)

# Force aspect ratio to be 'equal' so circles look like circles
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.6)

# Labels and Title
ax.set_title(f"Task 5: Realistic Flexible Link Animation\n(TV-LQR Tracking Perturbation)", fontsize=14)
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")

# --- Creating the Graphical Objects (Patches) ---
# Using 'Patches' instead of simple plot lines to draw solid shapes.
# These are initialized empty or at the origin and updated in the animation loop.

# Base Joint (Fixed) - Drawn once, stays at (0,0)
base_patch = patches.Circle((0, 0), joint_radius*1.2, fc=joint_color, ec=edge_color, zorder=5)
ax.add_patch(base_patch)

# Link 1 (Polygon/Rectangle)
# Initializing with dummy points, zorder < joints to appear "behind" them
link1_patch = patches.Polygon([[0,0],[0,0],[0,0],[0,0]], closed=True, fc=link_color, ec=edge_color, lw=2, zorder=3)
ax.add_patch(link1_patch)

# Joint 1 (Elbow Circle) - The moving joint between links
joint1_patch = patches.Circle((0, 0), joint_radius, fc=joint_color, ec=edge_color, zorder=5)
ax.add_patch(joint1_patch)

# Link 2 (Polygon/Rectangle)
link2_patch = patches.Polygon([[0,0],[0,0],[0,0],[0,0]], closed=True, fc=link_color, ec=edge_color, lw=2, zorder=3)
ax.add_patch(link2_patch)

# End-Effector Tip (Circle) - The tip of the robot
tip_patch = patches.Circle((0, 0), joint_radius*0.8, fc=joint_color, ec=edge_color, zorder=5)
ax.add_patch(tip_patch)

# Trace line (Red trail) - Shows the path history of the tip
trace, = ax.plot([], [], '-', lw=1, color='red', alpha=0.4, zorder=1)
trace_x, trace_y = [], []

# Text object to display current simulation time
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, 
                    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))

def init():
    r"""
    Initializes positions before animation starts.
    Called once at the beginning of FuncAnimation.
    """
    
    # Move everything out of view initially/reset data
    link1_patch.set_xy([[0,0],[0,0],[0,0],[0,0]])
    link2_patch.set_xy([[0,0],[0,0],[0,0],[0,0]])
    joint1_patch.center = (0,0)
    tip_patch.center = (0,0)
    trace.set_data([], [])
    time_text.set_text('')
    trace_x.clear()
    trace_y.clear()
    return link1_patch, joint1_patch, link2_patch, tip_patch, trace, time_text

def update(frame):
    r"""
    Updates the position and orientation of all patches for the current frame.
    Called repeatedly by FuncAnimation.
    
    Args:
        frame: The current time index from frames_indices
    """
    
    # Retrieving of the state vector at the current time step
    current_state = xx_sim[:, frame]
    
    # Calculating joint centers using Forward Kinematics
    p0, p1, p2 = forward_kinematics(current_state)
    
    # --- UPDATE LINK 1 ---
    # Calculating the 4 corners for the thick bar connecting Base (p0) and Elbow (p1)
    corners1 = get_thick_link_corners(p0, p1, link_width)
    link1_patch.set_xy(corners1) # Update polygon vertices
    
    # --- UPDATE JOINT 1 (Elbow) ---
    joint1_patch.center = tuple(p1) # Update circle center
    
    # --- UPDATE LINK 2 ---
    # Calculate the 4 corners for the thick bar connecting Elbow (p1) and Tip (p2)
    corners2 = get_thick_link_corners(p1, p2, link_width * 0.8) # Link 2 slightly thinner
    link2_patch.set_xy(corners2)
    
    # --- UPDATE TIP ---
    tip_patch.center = tuple(p2)
    
    # --- Update Trace & Time ---
    trace_x.append(p2[0])
    trace_y.append(p2[1])
    trace.set_data(trace_x, trace_y)
    
    time_text.set_text(f'Time = {frame*dt:.2f} s')
    
    return link1_patch, joint1_patch, link2_patch, tip_patch, trace, time_text

# --- Animation Configuration ---
# Defining a skip rate to speed up the animation visualization
skip = 5 
# Define the indices of frames to simulate
frames_indices = range(0, T_horizon, skip)

# Create the Animation Object
# interval: Delay between frames in milliseconds. 
# blit=True: Optimizes drawing by only re-drawing changed parts.
# IMPORTANT: if it doesen't work, use blit=False
anim = FuncAnimation(fig, update, frames=frames_indices,
                     init_func=init, blit=True, interval=dt*skip*1000, repeat=True)

#Saving the animation as .GIF
fps = int(1 / (dt * skip))  # it mantains the true velocity of the simulation
writer = PillowWriter(fps=fps)

# Saving the file in the correct folder
save_path = os.path.join(output_folder, "task5_animation.gif")
anim.save(save_path, writer=writer, dpi=120)
print(f"GIF saved as {save_path}")

plt.show()