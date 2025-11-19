# Linear System Control
## Controllability Gramians
* `CG.py` - Basic problem, has the option to do N-1 sum or Lyapunov Equatoin
* `CG_Bounded.py` - Same problem, but runs a small optimization problem to keep U within bounds. This is because
in a real-world problem, U would have to be within the bounds of the actuator. 
* `CG_AVL.py` - This is a much more difficult problem, where I tried to apply the same controllability discussion onto 
12-dimensional state vector.
## LQR Controller