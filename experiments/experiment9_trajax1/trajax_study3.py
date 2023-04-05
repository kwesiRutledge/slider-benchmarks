"""
trajax_study2.py
Description:
    In this file, I will use trajax to optimize the pusher-slider system's trajectory.
"""

import jax
import jax.numpy as jnp
import trajax
from trajax import optimizers

import sys
sys.path.append('../../src/python/')
from pusher_slider_sticking_force_input import PusherSliderStickingForceInputSystem

# Define Helper Functions

if __name__ == '__main__':
    # Constants
    N = 100
    x0 = jnp.array([-0.5, -0.5, jnp.pi/4])
    horizon = 1000
    n_u = 2

    nominal_scenario = {
        "obstacle_center_x": 0.0,
        "obstacle_center_y": 0.0,
        "obstacle_radius": 0.2,
    }
    ps = PusherSliderStickingForceInputSystem(
        nominal_scenario,
        dt = 0.01,
    )

    # Define helper functions
    @jax.jit
    def ps_cost(state: jnp.array, action: jnp.array, time_step: int) -> float:
        """
        simple_cost

        Description:
            Defines cost associated with each choice of state and action over all time_step's.

        Inputs:
            state - n-dimensional state vector
            action - d-dimensional action vector
        """

        # Constants
        theta = ps.theta
        x_star = ps.goal_point(theta)

        # Compute Distance to Target
        J_x = jnp.linalg.norm(state - x_star)

        # Compute Input Cost
        J_u = jnp.linalg.norm(action)
        # J_u = jnp.array(0.0)

        # TODO: Penalize actions outside of motion cone!

        return J_x.astype(float) + J_u.astype(float)


    @jax.jit
    def ps_dynamics(state: jnp.array, action: jnp.array, time_step: int) -> jnp.array:
        """
        simple_dynamics

        Description:
            Defines a 2d single integrator system's dynamics.

        Inputs:
            state - n-dimensional state vector
            action - d-dimensional action vector
        Outputs:
            x_next - n-dimensional state vector representing the derivative of the state
        """

        # Constants

        # Compute Dynamics
        x = state
        u = action

        x_dot = ps.closed_loop_dynamics(x, u)

        return x + ps.dt * x_dot

    # Run ilqr trajopt
    U0 = jnp.zeros((horizon, n_u))
    U0 = U0.at[:, 1].set(1.0)
    X, U, opt_obj, gradient, iteration = optimizers.scipy_minimize(
        ps_cost, ps_dynamics,
        x0,
        U0,
        method='Newton-CG',
    )
    if jnp.isclose(U, U0).all():
        print("Trajectory Optimization Failed")
    else:

        print("U = ", U)

    # print("iteration = ", iteration)
    # print("X = ", X)
    # print("opt_obj = ", opt_obj)
    #
    # print("ps.closed_loop_dynamics(X[0, :], U[0, :]) = ", ps.closed_loop_dynamics(X[0, :], U[0, :]))
    # print("x1 = x0 + ps.dt * ps.closed_loop_dynamics(X[0, :], U[0, :]) = ", X[0, :] + ps.dt * ps.closed_loop_dynamics(X[0, :], U[0, :]))
    # print("ps.closed_loop_dynamics(X[1, :], U[1, :]) = ", ps.closed_loop_dynamics(X[1, :], U[1, :]))
    #
    # print("ps.closed_loop_dynamics(X[0, :], [0, 10.0]) = ", ps.closed_loop_dynamics(X[0, :], jnp.array([0.0, 10.0])))
    # print("x1 = x0 + ps.dt * ps.closed_loop_dynamics(X[0, :], [0, 10.0]) = ",
    #       X[0, :] + ps.dt * ps.closed_loop_dynamics(X[0, :], jnp.array([0.0, 2.0])))
    # print("ps.closed_loop_dynamics(X[1, :], [0, 10.0]) = ", ps.closed_loop_dynamics(X[1, :], jnp.array([0.0, 10.0])))
    #
    # print(ps.goal_point(ps.theta))
    #
    # ps.control_affine_dynamics(x0)
    # print(ps.control_affine_dynamics(x0))

    # Plot Results
    X = X.T
    X = X.reshape((1, X.shape[0], X.shape[1]))
    U = U.T
    U = U.reshape((1, U.shape[0], U.shape[1]))
    th = ps.theta
    th = th.reshape((1, th.shape[0]))

    ps.save_animated_trajectory(
        x_trajectory=X,
        th=th,
        f_trajectory=U,
        hide_axes=False,
        filename="animate3.mp4",
        show_obstacle=True, show_goal=True,
    )