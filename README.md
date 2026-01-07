# OPTCON-RL_Project_Group_23
"Course Project #2 - Optimal Control of a Flexible Robotic Link".

# OPTCON-RL: Optimal Control of a Flexible Robotic Link
### Group 23 - Course Project #2

## Project Overview
This project involves the design of an optimal trajectory and control strategies for a flexible robotic link. The system is modeled as a double pendulum with nonlinear rotational stiffness. The goal is to perform trajectory generation, tracking via LQR and MPC, and animation of the results.

## Project Structure
* `src/`: Python source code for dynamics and control tasks.
* `report/`: LaTeX source files and final PDF report.
* `images/`: Generated plots and animation frames.
* `docs/`: Various helpfull documents 

## Tasks Breakdown

### Task 0: Problem Setup
* Discretization of dynamics (Forward Euler / Runge-Kutta 4th order).
* Implementation of the state-space equations using Symbolic differentiation (SymPy).

### Task 1: Trajectory Generation (I)
* Computation of equilibria using numerical root-finding.
* Optimal transition generation using Newton's algorithm for optimal control (Differential Dynamic Programming approach).
* Plots of the behaviour of states, inputs, Armijo line search, norm of the descent direction and cost along iterations

### Task 2: Trajectory Generation (II)
* Generation of a desired smooth state-input curve (Sigmoid transition).
* Application of the trajectory generation algorithm on this new curve for smoother behavior.
* Plots of the behaviour of states, inputs, Armijo line search, norm of the descent direction and cost along iterations

### Task 3: Trajectory Tracking via LQR
* Linearization about the generated trajectory.
* Solving the Time-Varying LQ Problem for optimal feedback control.

### Task 4: Trajectory Tracking via MPC
* Implementation of Model Predictive Control (MPC) to track the reference trajectory.
* Testing with perturbed initial conditions.

### Task 5: Animation
* Visualization of the system executing the tracking task.

## Parameters
The system parameters (masses $m_1, m_2$, lengths $l_1, l_2$, inertias $I_1, I_2$, stiffness $k$, etc.) are defined according to **Table 1** in the assignment specifications.

## Requirements

To run the simulation, you need **Python 3.x** installed along with the following libraries:

* **NumPy** (`numpy`): For numerical computations and linear algebra.
* **SciPy** (`scipy`): For optimization algorithms (`minimize`) and advanced integration.
* **Matplotlib** (`matplotlib`): For plotting trajectories and animations (`FuncAnimation`).
* **SymPy** (`sympy`): For symbolic derivation of the dynamics and Jacobians.
* **Control** (`control`): The Python Control Systems Library, used for Riccati equations (`dare`) and system analysis.
* **CVXPY** (`cvxpy`): For convex optimization problems (used in LQR/MPC solvers).

### Installation
You can install all the required dependencies using pip:

'''bash
pip install numpy scipy matplotlib sympy control cvxpy
'''

### Authors (Group 23)
* Alessandro Binci
* Alessandro Tampieri
* Lorenzo Tucci

---
*Reference: Optimal Control Course Project #2 - First Semester - Academic Year 2025/2026*
