"""
crafted1.py
Description:
    In this file, I will create my own custom pushing trajectory to use in a desired path for our pusher to follow.
"""

import jax
import jax.numpy as jnp
import trajax
from trajax import optimizers

import time
import yaml
from datetime import datetime

import sys
sys.path.append('../../src/python/')
from pusher_slider_sticking_force_input import PusherSliderStickingForceInputSystem

if __name__ == '__main__':
    # Constants

    data = {
        'horizon': 20, 'max_iter': 5000, 'num_samples': 500,
        'dt': 0.2, 'u0': 0.5, 'J_u_prefactor': 0.0, #0.0005,
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

    # Simulate using a dumb push
    u0 = jnp.array([0.0, data['u0']])

    x = x0
    x_k = x0

    u = u0
    for k in range(data['horizon']):
        # Update state
        x_kp1 = x_k + ps.dt * ps.closed_loop_dynamics(x_k, u0)

        # Update x and u histories
        x = jnp.vstack((x, x_kp1))
        u = jnp.vstack((u, u0))

        x_k = x_kp1 # Update state for next iteration

    # Plot
    X = x.T
    X = X.reshape((1, X.shape[0], X.shape[1]))
    U = u.T
    U = U.reshape((1, U.shape[0], U.shape[1]))
    th = ps.theta
    th = th.reshape((1, th.shape[0]))
    ps.save_animated_trajectory(
        x_trajectory=X,
        th=th,
        f_trajectory=U,
        hide_axes=False,
        filename="crafted-push1.mp4",
        show_obstacle=True, show_goal=True,
    )