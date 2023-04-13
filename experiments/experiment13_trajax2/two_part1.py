"""
two_part1.py
Description:
    In this file, I will use trajax to optimize the pusher-slider system's trajectory.
"""

import jax
import jax.numpy as jnp
import numpy
import trajax
from trajax import optimizers

import time
import yaml
from datetime import datetime

import sys
sys.path.append('../../src/python/')
from pusher_slider_sticking_force_input import PusherSliderStickingForceInputSystem

# Define Helper Functions

if __name__ == '__main__':
    # Constants

    data = {
        'horizon1': 80, 'horizon2': 40,
        'max_iter': 3500, 'num_samples': 5000,
        'elite_portion': 0.01,
        'dt': 0.05, 'u0': 0.5,
        'J_u_prefactor_diagonal': [0.005*(1/0.35), 0.005], #0.0005,
        'J_obstacle_prefactor': 10.0,
        'J_obstacle_prefactor_decay_rate': 0.1,
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

    # data['U_lb'] = [-ps.ps_cof * (data['u_max'] / 4), data['u_max'] / 4]
    # data['U_ub'] = [ps.ps_cof * (data['u_max'] / 4), data['u_max']]
    # This should not be right.
    # data['U_lb'] = [data['u_max'] / 4, -ps.ps_cof * (data['u_max'] / 4) ]
    # data['U_ub'] = [data['u_max']    , ps.ps_cof * (data['u_max'] / 4)]

    # This is even less right
    # data['U_lb'] = [0.0, -ps.ps_cof * data['u_max']]
    # data['U_ub'] = [data['u_max'], ps.ps_cof * data['u_max']]

    # Closer to being right
    data['U_lb'] = [-ps.ps_cof * data['u_max'], 0.0]
    data['U_ub'] = [ps.ps_cof * data['u_max'], data['u_max']]


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
        horizon = data['horizon1']

        # Compute Distance to Target
        J_x = jnp.linalg.norm(state[:2] - x_star[:2])

        # Compute Input Cost
        #J_u = (0.5 - (0.45*jnp.exp(time_step)/(jnp.exp(time_step) + jnp.exp(data['horizon']-time_step))))*jnp.linalg.norm(action)
        J_u = action.T @ jnp.diag(jnp.array(data['J_u_prefactor_diagonal'])) @ action
        # J_u = jnp.array(0.0)

        # TODO: Penalize actions outside of motion cone!
        #J_FC = jnp.max(0, jnp.abs(action.at[0].get()) - action.at[1].get()*st_cof)

        # Compute J_obstacle
        J_obstacle = data['J_obstacle_prefactor'] * jnp.abs(
            jnp.linalg.norm(state[:2]) - 1.5*ps.nominal_scenario['obstacle_radius']
        )
        # J_obstacle = \
        #     data['J_obstacle_prefactor'] * \
        #     jnp.exp(- 0.1*(time_step-(horizon/2))) * \
        #     jnp.abs(
        #         jnp.linalg.norm(state[:2]) - 1.5 * ps.nominal_scenario['obstacle_radius']
        #     )

        return J_x.astype(float) + J_u.astype(float) + J_obstacle.astype(float)


    @jax.jit
    def ps_cost2(state: jnp.array, action: jnp.array, time_step: int) -> float:
        """
        ps_cost2

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
        # J_u = (0.5 - (0.45*jnp.exp(time_step)/(jnp.exp(time_step) + jnp.exp(data['horizon']-time_step))))*jnp.linalg.norm(action)
        J_u = action.T @ jnp.diag(jnp.array(data['J_u_prefactor_diagonal'])) @ action
        # J_u = jnp.array(0.0)

        # TODO: Penalize actions outside of motion cone!
        # J_FC = jnp.max(0, jnp.abs(action.at[0].get()) - action.at[1].get()*st_cof)

        # Compute J_obstacle
        # J_obstacle = data['J_obstacle_prefactor'] * jnp.abs(
        #     jnp.linalg.norm(state[:2]) - 1.5*ps.nominal_scenario['obstacle_radius']
        # )

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
    U0 = jnp.zeros((data['horizon1'], data['n_u']))
    U0 = U0.at[:, 1].set(data['u0'])

    hyperparams = optimizers.default_cem_hyperparams()
    hyperparams['max_iter'] = data['max_iter']
    hyperparams['num_samples'] = data['num_samples']

    trajectory_optimization_start = time.time()
    X1, U0_star, opt_obj, = optimizers.cem(
        ps_cost, ps_dynamics,
        x0,
        U0,
        jnp.array(data['U_lb']),
        jnp.array(data['U_ub']),
        max_iter=hyperparams['max_iter'],
        num_samples=hyperparams['num_samples'],
        elite_portion=hyperparams['elite_portion'],
    )
    trajectory_optimization_end = time.time()
    data['trajopt_time1'] = trajectory_optimization_end - trajectory_optimization_start
    data['opt_obj1'] = float(opt_obj)

    if jnp.isclose(U0_star, U0).all():
        print("Trajectory Optimization Failed")
    else:

        print("U = ", U0_star)
        print("opt_obj = ", opt_obj)
        print("X = ", X1)

    # Run second part
    U1 = jnp.zeros((data['horizon2'], data['n_u']))
    U1 = U1.at[:, 0].set(data['u0'])

    x1 = X1[-1, :]

    trajectory_optimization_start = time.time()
    X2, U2_star, opt_obj, = optimizers.cem(
        ps_cost2, ps_dynamics,
        x1,
        U1,
        jnp.array(data['U_lb']),
        jnp.array(data['U_ub']),
        max_iter=hyperparams['max_iter'],
        num_samples=hyperparams['num_samples'],
        elite_portion=hyperparams['elite_portion'],
    )
    trajectory_optimization_end = time.time()
    data['trajopt_time2'] = trajectory_optimization_end - trajectory_optimization_start
    data['opt_obj2'] = float(opt_obj)

    if jnp.isclose(U2_star, U1).all():
        print("Trajectory Optimization Failed")
    else:

        print("U = ", U0_star)
        print("opt_obj = ", opt_obj)
        print("X = ", X2)

    # Save Full Trajectory
    X1 = X1.T
    X2 = X2.T
    X = jnp.hstack((X1, X2[:, 1:]))
    X = X.reshape((1, X.shape[0], X.shape[1]))

    U1 = U0_star.T
    U2 = U2_star.T
    U = jnp.hstack((U1, U2[:, 1:]))
    U = U.reshape((1, U.shape[0], U.shape[1]))

    data['X'] = numpy.asarray(X)
    data['U'] = numpy.asarray(U)

    # Save results
    now = datetime.now()
    d4 = now.strftime("%b-%d-%Y-%H:%M:%S")
    with open('data/trajax2_data_' + d4 + '.yml', 'w') as outfile:
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

    th = ps.theta
    th = th.reshape((1, th.shape[0]))

    ps.save_animated_trajectory(
        x_trajectory=X,
        th=th,
        f_trajectory=U,
        hide_axes=False,
        filename="trajax2.mp4",
        show_obstacle=True, show_goal=True,
    )