# Optimal Control and Reinforcement Learning Project 2025-2026
# Group 23: Alessandro Binci, Alessandro Tampieri, Lorenzo Tucci
# Task 5: Realistic Animation of the Flexible Link (TV-LQR Tracking)

import sys
import os

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches # To draw geometric forms
import dynamics as dyn
from ltv_solver_LQR import ltv_LQR
from matplotlib.animation import PillowWriter

#==============================================================================
# DATA GENERATION & SIMULATION
#==============================================================================

try:
    xx_opt_traj = np.load('optimal_traj_xx.npy')
    uu_opt_traj = np.load('optimal_traj_uu.npy')
except FileNotFoundError:
    print("Error: Optimal trajectory files (.npy) not found. Run Task 2 first.")
    sys.exit()

dt = dyn.dt
ns = dyn.ns
ni = dyn.ni
T_horizon = xx_opt_traj.shape[1]

AA_t = np.zeros((ns, ns, T_horizon))
BB_t = np.zeros((ns, ni, T_horizon))

print("Generating simulation data...")
for tt in range(T_horizon):
    x_t = xx_opt_traj[:, tt]
    u_t = uu_opt_traj[:, tt]
    AA_t[:, :, tt] = (dyn.dynamics_euler(x_t, u_t)[1]).T
    BB_t[:, :, tt] = (dyn.dynamics_euler(x_t, u_t)[2]).T

QQreg = np.diag([100, 100, 1, 1])
RRreg = np.diag([1])
QQreg_final = np.diag([100, 100, 1, 1])
SSreg = np.zeros((ni, ns))
KK = ltv_LQR(AA_t, BB_t, QQreg, RRreg, SSreg, QQreg_final, T_horizon, np.zeros(ns), np.zeros((ns,T_horizon)), np.zeros((ni,T_horizon)), np.zeros(ns))[0]

initial_perturbation = 0.25 # rad
xx_sim = np.zeros((ns, T_horizon))
uu_sim = np.zeros((ni, T_horizon))
xx_sim[:, 0] = xx_opt_traj[:, 0] + np.array([initial_perturbation, 0, 0, 0])

for tt in range(T_horizon - 1):
    error_t = xx_sim[:, tt] - xx_opt_traj[:, tt]
    uu_sim[:, tt] = uu_opt_traj[:, tt] + KK[:, :, tt] @ error_t
    xx_sim[:, tt + 1] = dyn.dynamics_euler(xx_sim[:, tt], uu_sim[:, tt])[0]
print("Data generation complete.")


#==============================================================================
# 2. ANIMATION SETUP
#==============================================================================

l1 = dyn.l_1
l2 = dyn.l_2

# --- Visual Styling Parameters (To match the CAD look) ---
link_width = 0.08       # Thickness of the rectangular bars
joint_radius = 0.06     # Radius of the circular joints
link_color = '#C0C0C0'  # Silver/Light Gray for links
joint_color = '#808080' # Darker Gray for joints
edge_color = 'black'    # Black outline for definition

# --- Helper Function for Forward Kinematics (Same as before) ---
def forward_kinematics(state):
    th1 = state[0]
    th2 = state[1]
    x1 = l1 * np.sin(th1)
    y1 = -l1 * np.cos(th1)
    x2 = x1 + l2 * np.sin(th1 + th2)
    y2 = y1 - l2 * np.cos(th1 + th2)
    return np.array([0,0]), np.array([x1, y1]), np.array([x2, y2])

# --- Helper Function for Geometry ---
def get_thick_link_corners(p_start, p_end, width):
    """
    Calculates the 4 corners of a thick rectangular bar connecting p_start and p_end.
    Used to draw the links as polygons.
    """
    # Vector along the link
    vec = p_end - p_start
    length = np.linalg.norm(vec)
    # Unit vector along the link
    unit_vec = vec / length
    # Perpendicular vector (normal)
    perp_vec = np.array([-unit_vec[1], unit_vec[0]])
    
    # Half width
    hw = width / 2.0
    
    # The 4 corners of the rectangle centered on the line segment
    c1 = p_start + perp_vec * hw
    c2 = p_start - perp_vec * hw
    c3 = p_end - perp_vec * hw
    c4 = p_end + perp_vec * hw
    return np.array([c1, c2, c3, c4])

# --- Figure Setup ---
fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
limit = (l1 + l2) * 1.2
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_title(f"Task 5: Realistic Flexible Link Animation\n(TV-LQR Tracking Perturbation)", fontsize=14)
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")

# --- Creating the Graphical Objects (Patches) ---
# These are initialized empty/at origin and updated in the loop.

# 1. Base Joint (Fixed)
base_patch = patches.Circle((0, 0), joint_radius*1.2, fc=joint_color, ec=edge_color, zorder=5)
ax.add_patch(base_patch)

# 2. Link 1 (Polygon/Rectangle)
# Initialize with dummy points, zorder < joints to appear "behind" them
link1_patch = patches.Polygon([[0,0],[0,0],[0,0],[0,0]], closed=True, fc=link_color, ec=edge_color, lw=2, zorder=3)
ax.add_patch(link1_patch)

# 3. Joint 1 (Elbow Circle)
joint1_patch = patches.Circle((0, 0), joint_radius, fc=joint_color, ec=edge_color, zorder=5)
ax.add_patch(joint1_patch)

# 4. Link 2 (Polygon/Rectangle)
link2_patch = patches.Polygon([[0,0],[0,0],[0,0],[0,0]], closed=True, fc=link_color, ec=edge_color, lw=2, zorder=3)
ax.add_patch(link2_patch)

# 5. End-Effector Tip (Circle)
tip_patch = patches.Circle((0, 0), joint_radius*0.8, fc=joint_color, ec=edge_color, zorder=5)
ax.add_patch(tip_patch)

# 6. Trace line (same as before, just for reference)
trace, = ax.plot([], [], '-', lw=1, color='red', alpha=0.4, zorder=1)
trace_x, trace_y = [], []

time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, 
                    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))

def init():
    """Initializes positions before animation starts."""
    # Move everything out of view initially
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
    """Updates the position and orientation of all patches."""
    current_state = xx_sim[:, frame]
    
    # Calculate joint centers
    p0, p1, p2 = forward_kinematics(current_state)
    
    # --- UPDATE LINK 1 ---
    # Calculate corners for the thick bar connecting p0 and p1
    corners1 = get_thick_link_corners(p0, p1, link_width)
    link1_patch.set_xy(corners1) # Update polygon vertices
    
    # --- UPDATE JOINT 1 (Elbow) ---
    joint1_patch.center = tuple(p1) # Update circle center
    
    # --- UPDATE LINK 2 ---
    # Calculate corners for the thick bar connecting p1 and p2
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
skip = 5 
frames_indices = range(0, T_horizon, skip)

print("Starting realistic animation...")
print("Close the plot window to terminate.")

# IMPORTANT: if it doesen't work, use blit=False
anim = FuncAnimation(fig, update, frames=frames_indices,
                     init_func=init, blit=True, interval=dt*skip*1000, repeat=True)

fps = int(1 / (dt * skip))  # oppure metti un valore tipo 30
writer = PillowWriter(fps=fps)

anim.save("task5_animation.gif", writer=writer, dpi=120)
print("GIF salvata come task5_animation.gif")

plt.show()