"""
trajax_study6.py
Description:
    In this file, I will use trajax to optimize the pusher-slider system's trajectory.
"""

import jax
import jax.numpy as jnp
import trajax
from trajax import optimizers

import time
import yaml
from datetime import datetime

import polytope as pc

import sys
sys.path.append('../../src/python/')
from pusher_slider_sticking_force_input import PusherSliderStickingForceInputSystem

# Define Helper Functions

if __name__ == '__main__':
    # Constants

    data = {
        'horizon': 300, 'max_iter': 100, 'num_samples': 50000,
        'dt': 0.05, 'u0': 3.0, 'J_u_prefactor': 0.0, #0.0005,
        'u_max': 20.0, 'x0': [-0.5, -0.5, jnp.pi/4],
    }
    x0 = jnp.array(data['x0'])

    nominal_scenario = {
        "obstacle_center_x": 0.0,
        "obstacle_center_y": 0.0,
        "obstacle_radius": 0.2,
    }
    ps = PusherSliderStickingForceInputSystem(
        nominal_scenario,
        dt=data['dt'],
    )
    data['n_u'] = ps.n_controls


    # Define helper functions
    @jax.jit
    def ps_cost(state: jnp.array, action: jnp.array, time_step: int) -> float:
        """
        ps_cost

        Description:
            Defines cost associated with each choice of state and action over all time_step's.

        Inputs:
            state - n-dimensional state vector
            action - d-dimensional action vector
        """

        # Constants
        theta = ps.theta
        st_cof = ps.st_cof
        x_star = ps.goal_point(theta)

        # Compute Distance to Target
        J_x = jnp.linalg.norm(state - x_star)

        # Compute Input Cost
        #J_u = (0.5 - (0.45*jnp.exp(time_step)/(jnp.exp(time_step) + jnp.exp(data['horizon']-time_step))))*jnp.linalg.norm(action)
        J_u = data['J_u_prefactor'] * jnp.linalg.norm(action)
        # J_u = jnp.array(0.0)

        # TODO: Penalize actions outside of motion cone!
        #J_FC = jnp.max(0, jnp.abs(action[0]) - action[1]*st_cof)

        # Compute J_obstacle
        J_obstacle = ps.nominal_scenario['obstacle_radius'] - jnp.linalg.norm(state[0:2])

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
    U0 = jnp.zeros((data['horizon'], data['n_u']))
    U0 = U0.at[:, 1].set(data['u0'])

    hyperparams = optimizers.default_cem_hyperparams()
    hyperparams['max_iter'] = data['max_iter']
    hyperparams['num_samples'] = data['num_samples']

    trajectory_optimization_start = time.time()
    X, U, opt_obj, = optimizers.cem_with_input_constraints(
        ps_cost, ps_dynamics,
        x0,
        U0,
        jnp.array(pc.extreme(ps.U)),
        max_iter=hyperparams['max_iter'],
        num_samples=hyperparams['num_samples'],
    )
    trajectory_optimization_end = time.time()
    data['trajopt_time'] = trajectory_optimization_end - trajectory_optimization_start
    data['opt_obj'] = float(opt_obj)

    if jnp.isclose(U, U0).all():
        print("Trajectory Optimization Failed")
    else:

        print("U = ", U)
        print("opt_obj = ", opt_obj)
        print("X = ", X)

    # Save results
    now = datetime.now()
    d6 = now.strftime("%b-%d-%Y-%H:%M:%S")
    with open('data/study6_data_' + d6 + '.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

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
        filename="animate4.mp4",
        show_obstacle=True, show_goal=True,
    )