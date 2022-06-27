# visuo-motor-adaptation-simulations


A set of codes to simulate LQG models with mismatched A and B in dynamics vs in observer.

# List of files:

* ```qian_4th_ordermodel.py```: Uses parameters from Qian et al 2013 for the dynamics of the system, and roughly tuned noise parameters (not estimated from empirical data) to create simulations of 1d trajectories with mismatched A and B. Creates figures right in the root.
* ```lqg_core.py```: main code needed to compute trajectories. Computes the L2 norm of a linear Ricatti Equations, and gives the optimal observer and controler matrices. Also has routines to simulate one step in a trajectory or a full trajectory.
* ```2nd_order.py```: A 2nd order dimensional system, simpler than Qian's.
