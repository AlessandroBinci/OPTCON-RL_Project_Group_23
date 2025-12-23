# OPTCON-RL_Project_Group_23
"Course Project #2 - Optimal Control of a Flexible Robotic Link".

# OPTCON-RL: Optimal Control of a Flexible Robotic Link
### Group 23 - Course Project #2

## Project Overview
This project involves the design of an optimal trajectory and control strategies for a flexible robotic link. [cite_start]The system is modeled as a double pendulum with nonlinear rotational stiffness[cite: 6]. The goal is to perform trajectory generation, tracking via LQR and MPC, and animation of the results.

## Project Structure
* `src/`: Python source code for dynamics and control tasks.
* `report/`: LaTeX source files and final PDF report.
* `images/`: Generated plots and animation frames.

## Tasks Breakdown

### Task 0: Problem Setup
* [cite_start]Discretization of dynamics (e.g., Runge-Kutta 4th order)[cite: 27, 30].
* Implementation of the state-space equations.

### Task 1: Trajectory Generation (I)
* [cite_start]Computation of two equilibria using numerical root-finding[cite: 38, 40].
* [cite_start]Optimal transition using Newton's algorithm for optimal control[cite: 39].

### Task 2: Trajectory Generation (II)
* [cite_start]Generation of a desired smooth state-input curve[cite: 52].
* Application of the trajectory generation algorithm on this new curve.

### Task 3: Trajectory Tracking via LQR
* [cite_start]Linearization about the generated trajectory[cite: 54].
* [cite_start]Solving the LQ Problem for optimal feedback control[cite: 56].

### Task 4: Trajectory Tracking via MPC
* [cite_start]Implementation of Model Predictive Control (MPC) to track the reference trajectory[cite: 63, 64].
* [cite_start]Testing with perturbed initial conditions[cite: 65].

### Task 5: Animation
* [cite_start]Visualization of the system executing the tracking task[cite: 66].

## Parameters
[cite_start]The system parameters (masses $m_1, m_2$, lengths $l_1, l_2$, inertias $I_1, I_2$, stiffness $k$, etc.) are defined according to **Table 1** in the assignment specifications[cite: 26].

## Requirements
* Python 3.x
* NumPy, SciPy, Matplotlib
* (Optional) [cite_start]CasADi or Symbolic toolboxes for Jacobians[cite: 36].

## Authors (Group 23)
* Student 1
* Student 2
* Student 3

---
[cite_start]*Reference: Optimal Control Course Project #2 - November 28, 2025 [cite: 1, 3]*
