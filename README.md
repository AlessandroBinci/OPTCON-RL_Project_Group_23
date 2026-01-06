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

## Tasks Breakdown

### Task 0: Problem Setup
* Discretization of dynamics (e.g., Runge-Kutta 4th order).
* Implementation of the state-space equations.

### Task 1: Trajectory Generation (I)
* Computation of two equilibria using numerical root-finding.
* Optimal transition using Newton's algorithm for optimal control.

### Task 2: Trajectory Generation (II)
* Generation of a desired smooth state-input curve.
* Application of the trajectory generation algorithm on this new curve.

### Task 3: Trajectory Tracking via LQR
* Linearization about the generated trajectory.
* Solving the LQ Problem for optimal feedback control.

### Task 4: Trajectory Tracking via MPC
* Implementation of Model Predictive Control (MPC) to track the reference trajectory.
* Testing with perturbed initial conditions.

### Task 5: Animation
* Visualization of the system executing the tracking task.

## Parameters
The system parameters (masses $m_1, m_2$, lengths $l_1, l_2$, inertias $I_1, I_2$, stiffness $k$, etc.) are defined according to **Table 1** in the assignment specifications.

## Requirements
* Python 3.x
* NumPy, SciPy, Matplotlib
* CasADi or Symbolic toolboxes for Jacobians
* ... da continuare

## Authors (Group 23)
* Alessandro Binci
* Alessandro Tampieri
* Lorenzo Tucci

---
*Reference: Optimal Control Course Project #2 - First Semester - Academic Year 2025/2026*
