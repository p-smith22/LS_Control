"""
Driver function that defines the system

Takes a continuous system
x_dot = Ax + Bu
y = Cx
and discretizes such that
x_{k+1} = Ax_{k} + Bu_{k}
y_{k} = Cx_{k}

Therefore, for nw systems, simply change the definition of the problem
to your cts system and the rest will adjust automatically
"""
