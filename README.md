# Linear System Control
## Controllability Gramians - `CG` Directory

* `CG.py` - Basic problem, has the option to do N-1 sum or Lyapunov Equatio
* `CG_Bounded.py` - Same problem, but runs a small optimization problem to keep U within bounds. This is because
in a real-world problem, U would have to be within the bounds of the actuator. 
* `CG_AVL.py` - This is a more difficult problem, where the system had to be transferred from continuous to discrete. 
Additionally, the state vector is 12 components, so a much higher dimensionality problem. 

## LQR Controller - `LQR` Directory

* `Cart_Pole.py` - Simple cart pole problem that uses ARE to solve for optimal controls.

## MPC Controller - `MPC` Directory

* `driver_SD.py` - Simple spring dashpot mass system. 
* `driver_AVL.py` - AVL example using MPC controller.
* `NL_Drag.py` - A nonlinear drag model, which demonstrates how a nonlinear model would have to be linearized at each 
step prior to MPC. 
* `src` Directory - Contains the utils and background functions for MPC.
