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
# Add 'docs/Task1' folder path to store the images
output_folder = os.path.join(current_dir, 'images', 'Task1')
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


#--------------- TESTING EQUILIBRIUM POINTS -------------------
# Checking if the equilibrium points are actually equilibrium points
# We use the dynamics function (only with the eq. points 1 and 2)...
# ... to test if in 10 sequential time instants the value of the ... 
# ... state doesn't change (it remains at the equilibrium)

# xx_eq1, xx_eq2, uu_eq1,uu_eq2 = eq.eq_gen(th1_init, tau1_des, th1_final)
# for jj in range(10):
   
#     xx_try = np.zeros(ns)
#     uu_try = np.zeros(ni)
#     xx_atamp = np.zeros((ns,jj))
#     xx_try = xx_eq2 #change to xx_eq1 to test the eq. point nr.1
#     uu_try = uu_eq2 #change to uu_eq1 to test the eq. point nr.1

#     xx_atamp = dyn.dynamics_euler(xx_try, uu_try)[0]
#     print(xx_atamp)

#-------------------------------------------------------------------

# Computing the reference trajectory
xx_ref, uu_ref = ref_traj.gen(tf, dt, ns, ni, th1_init, tau1_des, th1_final)

# Defining the initial guess
xx = np.zeros((ns, TT))
uu = np.zeros((ni, TT))

for tt in range(TT):
    xx[:, tt] = xx_ref[:, 0]  # Setting the starting point as the eq point 1
    uu[:, tt] = uu_ref[:, 0]  

# Defining the number of max iterations
max_iters = 20

# Defining the matrices
QQ = cst.QQ
RR = cst.RR
SS = np.zeros((ni, ns))

# Variables required for plots
xx_history = []
uu_history = []

# List to store the Descent Directions (Delta x, Delta u) (required later for plots)
dx_history = []
du_history = []
# List to store the costs (required later for plots)
cost_history = [] 
# List to store data of Armijo plot (Line Search) (required later for plots)
global_armijo_data = []

# Newton's method
for kk in range(max_iters):
    AA_kk = np.zeros((ns, ns, TT))          # initialization of matrices
    BB_kk = np.zeros((ns, ni, TT))
    qq_kk = np.zeros((ns, TT))
    rr_kk = np.zeros((ni, TT))
    cost_current = 0                     # inizialization of the cost
    xx_history.append(xx.copy())         # saving the state at each iteration
    uu_history.append(uu.copy())         # saving the input at each iteration
    
    # Computing the matrices for the solver and the stage cost
    for tt in range(TT-1):
        AA_kk[:,:, tt] = (dyn.dynamics_euler(xx[:, tt], uu[:, tt]) [1]).T
        BB_kk[:,:, tt] = (dyn.dynamics_euler(xx[:, tt], uu[:, tt]) [2]).T
        cost_actual = cst.stage_cost(xx[:,tt],uu[:,tt],xx_ref[:,tt],uu_ref[:,tt]) [0]
        qq_kk[:,tt] = cst.stage_cost(xx[:,tt],uu[:,tt],xx_ref[:,tt],uu_ref[:,tt]) [1]
        rr_kk[:,tt] = cst.stage_cost(xx[:,tt],uu[:,tt],xx_ref[:,tt],uu_ref[:,tt]) [2]
        cost_current += cost_actual 

    # Computing the terminal cost and adding it to the total cost
    termcost_actual = cst.termcost(xx[:,-1], xx_ref[:,-1],QQ) [0]   
    qqT_kk = cst.termcost(xx[:,-1], xx_ref[:,-1],QQ) [1]
    cost_current += termcost_actual
    cost_history.append(cost_current) # saving the cost at each iteration

    # Getting the gain matrix and the feedforward term from the ltv_LQR solver
    KK, sigma = ltv_LQR(AA_kk, BB_kk, QQ, RR, SS, QQ, TT, np.zeros((ns)), qq_kk, rr_kk, qqT_kk) [0:2]
    # Getting the descent directions from the ltv_LQR solver
    dx_lin , du_lin = ltv_LQR(AA_kk, BB_kk, QQ, RR, SS, QQ, TT, np.zeros((ns)), qq_kk, rr_kk, qqT_kk) [3:]

    ###----- Armijo -----###

    # Setting the Armijo's parameters
    gamma = 1
    beta = 0.7
    cc = 0.5
    max_armijo_iters = 10
    ii = 1
    slope = 0
    
    # Computing the slope for the Armijo loop
    for tt in range(TT):

        # Dot Product of Gradient_x * Delta_x_lin
        slope += np.dot(qq_kk[:, tt], dx_lin[:, tt])
    
        # If we are not in the last step, we add Gradient_u * Delta_u_lin (because in the last time instant T, uu isn't defined)
        if tt < TT - 1:
            slope += np.dot(rr_kk[:, tt], du_lin[:, tt])

    # Adding slope term for the final time T (Gradient_x_T * Delta_x_T)
    slope += np.dot(qqT_kk, dx_lin[:, -1])


    # ---------------------------------------------------------
    # Starting generating data for Armijo plots (Line Search)
    # ---------------------------------------------------------
    iters_to_debug = [0, 1, 2, 5] # iteration samples of which we'll plot armijo
    armijo_data_iter = {}  # dictionary to store iterations' data

    # Computing Line Search curve (Feedback) only for iteration {kk} equal to 0,1,5
    if kk in iters_to_debug:

        print(f"Computing Line Search curve (Feedback) only for iteration {kk}")
        
        # Defining X-axis (20 points between 0 and 1.0)
        gammas_test = np.linspace(0, 1.0, 100) 
        # Initializing lists to store later the real cost and the armijo threshold line
        costs_test = []
        armijo_thresholds = []
        linear_approximations = []
        
        # Simulating the exact cost that the algorithm would see
        for g_val in gammas_test:
            
            # Exact reply of the while loop update logic
            # x_roll changes and the input adapts due to the feedback K
            
            x_roll = xx[:, 0].copy() # initial state copy
            # Temporary vector used to store the whole trajectory test
            xx_roll_traj = np.zeros((ns, TT))
            xx_roll_traj[:, 0] = x_roll
            
            c_cum = 0
            
            for t in range(TT - 1):
                # Computing delta_x wrt actual nominal trajectory 'xx'
                dx_curr = x_roll - xx[:, t]
                
                # Updating feedback rule (same of while loop)
                # u = u_nom + gamma * sigma + K * dx
                ff_term = g_val * sigma[:, t]
                fb_term = KK[:, :, t] @ dx_curr
                u_applied = uu[:, t] + ff_term + fb_term
                
                # Stage cost
                c_cum += cst.stage_cost(x_roll, u_applied, xx_ref[:, t], uu_ref[:, t])[0]
                
                # Dynamics
                x_roll = dyn.dynamics_euler(x_roll, u_applied)[0]
                xx_roll_traj[:, t+1] = x_roll # Salviamo per il passo dopo
            
            # Final cost
            c_cum += cst.termcost(x_roll, xx_ref[:, -1], QQ)[0]
            
            costs_test.append(c_cum) # Saving the cost
            
            # Armijo line
            thresh = cost_current + cc * g_val * slope 
            armijo_thresholds.append(thresh)

            # Tangent line
            lin_approx = cost_current + 1.0 * g_val * slope 
            linear_approximations.append(lin_approx)

            

        # Saving curve data
        armijo_data_iter['iter'] = kk
        armijo_data_iter['gammas'] = gammas_test
        armijo_data_iter['real_costs'] = costs_test
        armijo_data_iter['thresholds'] = armijo_thresholds
        armijo_data_iter['linear_approx'] = linear_approximations
        
        # Placeholder for tested points
        armijo_data_iter['tested_gammas'] = []
        armijo_data_iter['tested_costs'] = []
    # ---------------------------------------------------------

    # Temporary lists to store Armijo stepsizes (Orange points)
    temp_tested_gammas = []
    temp_tested_costs = []

    # Real Armijo loop
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

        # Saving the stepsize attempt (Orange point) ---
        temp_tested_gammas.append(gamma)
        temp_tested_costs.append(cost_temp)
       
        if cost_temp >= cost_current  + cc*gamma*slope:
                  
                  # Updating the stepsize
                  gamma = beta* gamma
                  ii += 1
        else:

            # Saving the accepted point
            if kk in iters_to_debug and 'armijo_data_iter' in locals() and armijo_data_iter:
                armijo_data_iter['accepted_gamma'] = gamma
                armijo_data_iter['accepted_cost'] = cost_temp

                # Saving the history of the attempts
                armijo_data_iter['tested_gammas'] = temp_tested_gammas
                armijo_data_iter['tested_costs'] = temp_tested_costs
                global_armijo_data.append(armijo_data_iter)
            
            xx = xx_temp.copy()
            uu = uu_temp.copy()

            break

    

    # Saving the computed descent direction
    dx_history.append(dx_lin.copy())
    du_history.append(du_lin.copy())

xx_history.append(xx.copy())
uu_history.append(uu.copy())

# --- PLOT 1: Optimal Trajectory vs Desired Curve ---
# Visualization of the optimal trajectories of states x1, x2 and the input wrt their own desired reference curves

# Temporal axis definition
# We assume that 'tf' and 'TT' are defined in our main
time_axis = np.linspace(0, tf, TT)

# Figure creation
plt.figure(figsize=(10, 8))

# --- Subplot 1: First State (Theta 1) ---
plt.subplot(3, 1, 1)
# Desired traj. - Black Dashed Line
plt.plot(time_axis, xx_ref[0, :], 'k--', linewidth=2, label=r'$\theta_{1,des}$ (Desired)')
# Optimal traj. - Solid Colored Line
plt.plot(time_axis, xx[0, :], 'b-', linewidth=2, label=r'$\theta_{1,opt}$ (Optimal)')
plt.ylabel(r'$\theta_1$ [rad]')
plt.title('Optimal Trajectory vs Desired Curve')
plt.grid(True)
plt.legend(loc='best')

# --- Subplot 2: Second State (Theta 2) ---
plt.subplot(3, 1, 2)
# Desired traj.
plt.plot(time_axis, xx_ref[1, :], 'k--', linewidth=2, label=r'$\theta_{2,des}$ (Desired)')
# Optimal traj.
plt.plot(time_axis, xx[1, :], 'r-', linewidth=2, label=r'$\theta_{2,opt}$ (Optimal)')
plt.ylabel(r'$\theta_2$ [rad]')
plt.grid(True)
plt.legend(loc='best')

# --- Subplot 3: Input ---
plt.subplot(3, 1, 3)
# Desired traj.
plt.plot(time_axis, uu_ref[0, :], 'k--', linewidth=2, label=r'$u_{des}$ (Desired)')
# Optimal traj.
plt.plot(time_axis, uu[0, :], 'g-', linewidth=2, label=r'$u_{opt}$ (Optimal)')
plt.ylabel(r'Torque [Nm]')
plt.xlabel(r'Time [s]')
plt.grid(True)
plt.legend(loc='best')

# Optimization of the spaces and storage
plt.tight_layout()

# Save the image in the correct folder
plt.savefig(os.path.join(output_folder, 'Task1_Optimal_vs_Desired.png'), dpi=300)
# Print on screen
plt.show()

# --- PLOT 1-bis: Optimal Velocities vs Desired (Zero) ---
# Visualization of the states x3 (Velocity Theta 1) and x4 (Velocity Theta 2)

plt.figure(figsize=(10, 8))

# --- Subplot 1: Third State (x3 - Velocity Theta 1) ---
plt.subplot(2, 1, 1)
# Desired traj. (generally zero)
plt.plot(time_axis, xx_ref[2, :], 'k--', linewidth=2, label=r'$\dot{\theta}_{1,des}$ (Desired)')
# Optimal traj.
plt.plot(time_axis, xx[2, :], 'b-', linewidth=2, label=r'$\dot{\theta}_{1,opt}$ (Optimal)')
plt.ylabel(r'$\dot{\theta}_1$ [rad/s]')
plt.title('Optimal Velocities vs Desired')
plt.grid(True)
plt.legend(loc='best')

# --- Subplot 2: Fourth State (x4 - Velocity Theta 2) ---
plt.subplot(2, 1, 2)
# Desired traj. (generally zero)
plt.plot(time_axis, xx_ref[3, :], 'k--', linewidth=2, label=r'$\dot{\theta}_{2,des}$ (Desired)')
# Optimal traj.
plt.plot(time_axis, xx[3, :], 'r-', linewidth=2, label=r'$\dot{\theta}_{2,opt}$ (Optimal)')
plt.ylabel(r'$\dot{\theta}_2$ [rad/s]')
plt.xlabel(r'Time [s]')
plt.grid(True)
plt.legend(loc='best')

plt.tight_layout()
plt.savefig(os.path.join(output_folder,'Task1_Velocities.png'), dpi=300)
plt.show()

# --- PLOT 2: Optimal trajectory (states x1,x2 and u), desired curve and few intermediate trajectories ---
# Visualization of the optimal trajectories of states x1, x2 and the input u at some intermediate iterations

# Define time axis
time_axis = np.linspace(0, tf, TT)

# Select specific iterations to plot: [0, 1, 2, Final]
total_iters = len(xx_history)
desired_indices = [0, 1, 2, total_iters - 1]

# Filter indices to ensure they exist and avoid duplicates
indices_to_plot = []
for idx in desired_indices:
    if idx < total_iters:
        if idx not in indices_to_plot:
            indices_to_plot.append(idx)

print(f"Plotting iterations: {indices_to_plot}")

# Retrieve trajectory data
xx_plot_list = [xx_history[i] for i in indices_to_plot]
uu_plot_list = [uu_history[i] for i in indices_to_plot]
num_plots = len(indices_to_plot)

# --- MANUAL COLOR MANAGEMENT ---
# Here we can define manually the color for each curve which will be plotted.
# The order corresponds to: [Iter 0, Iter 1, Iter 2, Final] 
# The following names can be used: ('orange','blue','green','red')

custom_colors = [
    'tab:orange',  # Iter 0 (Initial Guess)
    'tab:blue',    # Iter 1
    'tab:green',   # Iter 2
    'tab:red'      # Final (Optimal)
]

# Safety: in the case we have less iterations, we truncate the color list
if len(custom_colors) > num_plots:
    current_colors = custom_colors[:num_plots]
else:
    # If there are no iterations, we repeat the last one
    current_colors = custom_colors + [custom_colors[-1]]*(num_plots-len(custom_colors))

LW = 1.5 # Line Width

# --- FIGURE 1: Positions and Input ---
plt.figure(figsize=(12, 10))

# Subplot 1: Theta 1 Evolution
plt.subplot(3, 1, 1)
# Reference (Dashed Black)
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
plt.savefig(os.path.join(output_folder,'Task1_Evolution_Pos_Input_ManualColors.png'), dpi=300)
plt.show()


# --- PLOT 2-bis: Optimal trajectory (states x3 and x4), desired curve and few intermediate trajectories ---
# Visualization of the optimal trajectories of states x3 and x4 (velocities) at some intermediate iterations

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
plt.savefig(os.path.join(output_folder,'Task1_Evolution_Velocities_ManualColors.png'), dpi=300)
plt.show()

# --- PLOT 3: Armijo Line Search (Updated with Tested Steps) ---
# Visualization of Armijo Line Search

if len(global_armijo_data) > 0:
    for data in global_armijo_data:
        iter_idx = data['iter']
        gammas = data['gammas']
        real_costs = data['real_costs']
        lines = data['thresholds']
        tangents = data.get('linear_approx')
        
        acc_g = data['accepted_gamma']
        acc_c = data['accepted_cost']
        
        # Recover tested points (the orange ones)
        tested_g = data.get('tested_gammas', [])
        tested_c = data.get('tested_costs', [])
        
        plt.figure(figsize=(10, 6))
        
        # 1. Real cost curve
        plt.plot(gammas, real_costs, 'g-', linewidth=2, label='Real Cost (Closed Loop)')

       # 2. Linear Approximation (Tangent) - Red Dashed
        if tangents is not None:
             plt.plot(gammas, tangents, 'r--', linewidth=1.5, label=r'Linear Approx ($\nabla J^T d$)')

        # 3. Armijo Threshold - Green Dashed (Thinner)
        plt.plot(gammas, lines, 'g-.', linewidth=1, alpha=0.7, label=r'Armijo Threshold ($c \cdot \nabla J^T d$)')

        # 4. Tested Points (Orange Scatter)
        if len(tested_g) > 0:
            plt.scatter(tested_g, tested_c, color='orange', s=50, zorder=5, label='Tested Steps')
        
        # 5. Accepted Point (Red Scatter)
        if acc_g is not None:
            plt.scatter(acc_g, acc_c, color='red', s=100, zorder=6, label=fr'Accepted $\gamma={acc_g:.4f}$')
        

            
        plt.title(f'Newton Descent with Armijo (Iter {iter_idx})')
        plt.xlabel(r'Step Size $\gamma$')
        plt.ylabel(r'Cost $J$')
        plt.grid(True)
        plt.legend(loc='best')
        
        # --- Zoom to see the tangent line ---
        
        # Determining the plot bounds using only the real cost curve
        valid_costs = list(real_costs)
        valid_costs.append(cost_current)
        
        # Filtering possible values as NaN o inf 
        valid_costs = [c for c in valid_costs if np.isfinite(c)]
        
        if len(valid_costs) > 0:
            y_data_min = min(valid_costs)
            y_data_max = max(valid_costs)
            
            # Computing the range variation of real cost values
            y_span = y_data_max - y_data_min
            if y_span == 0: y_span = 1.0 # to avoid crashes on flatten plots
            
            # Defining margins
            #    - Lower margin: 10% of range variation (so it doesn't touch the ground)
            #    - Upper margin: 20% of range variation (so it's wider, to see better the tangent line)
            margin_bottom = 0.1 * y_span
            margin_top = 0.2 * y_span
            
            # If we accept a point that is very low (lower than the tested minimum), we enlarge the space at the end
            if acc_c is not None and acc_c < y_data_min:
                y_data_min = acc_c
            
            # Defining bounds for Y
            plt.ylim(y_data_min - margin_bottom, y_data_max + margin_top)
            
            # Defining bounds for X
            max_x = max(gammas) if len(gammas) > 0 else 1.2
            plt.xlim(-0.05, max_x * 1.05)   # We start a little bit earlier than the 0 (i.e. -0.05) to better see y-axis and the intercept line

        plt.savefig(os.path.join(output_folder,f'Task1_Armijo_LineSearch_Iter_{iter_idx}.png'), dpi=300)
        plt.show()

# --- PLOT 4: Norm of the Descent Direction (Semi-Log Scale) ---
# Visualization of the Norm of the descent direction along iterations (semi-logarithmic scale)

if len(dx_history) > 0 and len(du_history) > 0:
    
    descent_norms = []
    iterations = []
    
    # Computation of the norm for each stored iteration
    for i in range(len(dx_history)):
        dx_traj = dx_history[i] # Matrix (ns x TT)
        du_traj = du_history[i] # Matrix (ni x TT)
        
        # Combine in one single vector to compute the "norm of the total perturbation"
        # Use the norm L2 (Euclidian) of all flattened values 
        # Norm = sqrt( sum(dx^2) + sum(du^2) )
        vector_dx = dx_traj.flatten()
        vector_du = du_traj.flatten()
        
        total_norm = np.linalg.norm(np.concatenate((vector_dx, vector_du)))
        
        descent_norms.append(total_norm)
        iterations.append(i)

    # Creation of the plot
    plt.figure(figsize=(10, 6))
    
    # Semi-logaritmic plot (logaritmic Y axis)
    plt.semilogy(iterations, descent_norms, 'bo-', linewidth=2, markersize=6, label=r'$||\Delta z||$ (Descent Direction)')
    
    plt.title('Norm of Descent Direction along Iterations')
    plt.xlabel('Iteration $k$')
    plt.ylabel(r'Norm $||\Delta z^k|| = \sqrt{||\Delta x^k||^2 + ||\Delta u^k||^2}$')
    plt.grid(True, which="both", axis="both", ls="--", alpha=0.6) # Specific grid for log plot
    plt.legend(loc='best')
    
    plt.savefig(os.path.join(output_folder,'Task1_Descent_Direction_Norm.png'), dpi=300)
    plt.show()

else:
    print("No stored history of the directions (dx_history empty).")


# --- PLOT 5: Cost along iterations (semi-logarithmic scale) ---
# Visualization of the Cost along iterations (semi-logarithmic scale)

if len(cost_history) > 0:
    plt.figure(figsize=(10, 8))
    
    # Using of 'semilogy' for the logaritmic scale on Y
    # marker='o' for the dots on the plot
    plt.semilogy(range(len(cost_history)), cost_history, 
                 color='tab:cyan', marker='o', linestyle='-', 
                 linewidth=2, markersize=8)

    plt.title("Cost along iterations (log scale)", fontsize=16)
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Cost", fontsize=14)
    
    # Specific grid for log plot (both for major and minor lines)
    plt.grid(True, which="major", linestyle='-', linewidth=0.8)
    plt.grid(True, which="minor", linestyle=':', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder,'Task1_Cost_LogScale.png'), dpi=300)
    plt.show()
else:
    print("No stored history of the costs.")