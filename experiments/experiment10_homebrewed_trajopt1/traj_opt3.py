"""
traj_opt3.py
Description:
    This script will run a trajectory optimizer on the simple force-based pusher slider and then visualize the found trajectory.

"""

import sys, time
from datetime import datetime

import matplotlib.pyplot as plt
import jax.numpy as jnp

import yaml

sys.path.append('../../')
#from src.python.pusher_slider import PusherSliderStickingVelocityInputSystem
from src.python.pusher_slider_sticking_force_input import PusherSliderStickingForceInputSystem

from src.python.simple_traj_opt import ic_traj_opt

# Functions
# =========
def NStepCompositionFunction(u: jnp.array, horizon: int, n_controls: int, dynamics_function, dt: float):
    """
    NStepCompositionFunction
    Description:
        Computes the N-step composition of the closed loop dynamics.

    """
    # Reshape According to input dimension of ps
    input_dim = 2

    # Compute Composition
    x_t = x0
    for k in range(horizon):
        u_t = u[k, :].reshape((n_controls,))
        x_tp1 = x_t + dynamics_function(x_t, u_t) * dt

        # Set new variable values for next loop
        x_t = x_tp1

    return x_t

# Constants

# x0 = jnp.array([
#     [1.0], [1.0], [jnp.pi / 3], [0.0]
# ])
traj_opt_results = {
    'u_step_size': 1000.0, 'N_points': 20,
    'num_traj_opt_iters': 20,
    'dt': 0.05,
    'x0': [-0.1, -0.1, jnp.pi / 3],
}

nominal_scenario = {
    "obstacle_center_x": 0.0,
    "obstacle_center_y": 0.0,
    "obstacle_radius": 0.2,
}
ps = PusherSliderStickingForceInputSystem(
    nominal_scenario=nominal_scenario,
    dt=traj_opt_results['dt'],
)

# [ [ -0.1, -0.1, pi/3, 0.02 ]' , [ -0.1, -0.1, pi/6, 0.02 ]' , [ -0.2, -0.1, pi/6, 0.02 ]' , [ -0.2, -0.1, 0, 0.02 ]' ]
opt_times = []
N_repetition = 1
x0 = jnp.array(traj_opt_results['x0'])

for rep_index in range(N_repetition):
    u_opt, opt_time, final_loss = ps.ic_traj_opt(
        x0,
        horizon=traj_opt_results['N_points'],
        num_traj_opt_iters=traj_opt_results['num_traj_opt_iters'],
        u_step_size=traj_opt_results["u_step_size"],
    )
    traj_opt_results["opt_time"] = opt_time
traj_opt_results["u_opt"] = u_opt

traj_opt_results["dt"] = ps.dt

print('Times taken for trajectory optimization:')
print(opt_times)
print(traj_opt_results)

fig = plt.figure()
ax = fig.add_subplot(111)
tempElts = ps.plot_single(
    x0, ps.theta,
    ax,
    limits=[[-1.0, 1.0], [-1.0, 1.0]],
)
tempElts = ps.plot_single(
    NStepCompositionFunction(
        u_opt,
        traj_opt_results['N_points'],
        ps.n_controls, ps.closed_loop_dynamics,
        ps.dt,
    ),
    ps.theta,
    ax,
    limits=[[-1.0, 1.0], [-1.0, 1.0]],
    show_obstacle=False,
    show_goal=False,
)

fig.savefig('data/traj_opt3.png')

now = datetime.now()
d4 = now.strftime("%b-%d-%Y-%H:%M:%S")
with open('data/traj_opt3_data_' + d4 + '.yml', 'w') as outfile:
    yaml.dump(traj_opt_results, outfile, default_flow_style=False)
