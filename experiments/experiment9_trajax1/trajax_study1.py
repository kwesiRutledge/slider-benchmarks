"""
trajax_study1.py
Description:
    In this file, I will use trajax to optimize a simple system's trajectory.
"""

import jax
import jax.numpy as jnp
import trajax
from trajax import optimizers

# Define Helper Functions
@jax.jit
def simple_cost(state: jnp.array, action: jnp.array, time_step: int)->float:
    """
    simple_cost

    Description:
        Defines cost associated with each choice of state and action over all time_step's.

    Inputs:
        state - n-dimensional state vector
        action - d-dimensional action vector
    """

    # Constants
    x_star = jnp.array([1.0, 1.0])

    # Compute Distance to Target
    J_x = jnp.linalg.norm(state - x_star)

    # Compute Input Cost
    J_u = jnp.linalg.norm(action)

    return J_x.astype(float) + J_u.astype(float)

@jax.jit
def simple_dynamics(state: jnp.array, action: jnp.array, time_step: int)->jnp.array:
    """
    simple_dynamics

    Description:
        Defines a 2d single integrator system's dynamics.

    Inputs:
        state - n-dimensional state vector
        action - d-dimensional action vector
    Outputs:
        x_dot - n-dimensional state vector representing the derivative of the state
    """

    # Constants
    dt = 0.1

    # Compute Dynamics
    x = state
    u = action
    x_dot = u

    return x + dt * x_dot


if __name__ == '__main__':
    # Constants
    N = 100
    x0 = jnp.array([-0.5, 0.2])
    horizon = 50
    dt = 0.1
    n_u = 2

    # Run ilqr trajopt
    X, U, opt_obj, gradient, adjoints, lqr, iteration = optimizers.ilqr(
        simple_cost,
        simple_dynamics,
        x0,
        jnp.ones((horizon, n_u)),
        maxiter=1000,
    )
    if jnp.isclose(U, jnp.zeros((horizon, n_u))).all():
        print("Trajectory Optimization Failed")
    else:
        print("U = ", U)

    print("iteration = ", iteration)
    print("X = ", X)
    print("opt_obj = ", opt_obj)