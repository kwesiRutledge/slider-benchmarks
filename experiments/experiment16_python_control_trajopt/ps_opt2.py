"""
ps_opt2.py
Description:
    This script is a simple test of the python-control trajectory optimizer
    when applied to the pusher-slider system!
"""

import numpy as np
import control as ct
import control.optimal as opt
import matplotlib.pyplot as plt

import polytope as pc
import scipy.optimize as optimize

import sys
sys.path.append('../../src/python/')
from pusher_slider_sticking_force_input import PusherSliderStickingForceInputSystem

from datetime import datetime
import yaml
import jax.numpy as jnp

# Constants

# Algorithm

def vehicle_update(t, x, u, params):
    # Get the parameters for the model
    l = params.get('wheelbase', 3.)         # vehicle wheelbase
    phimax = params.get('maxsteer', 0.5)    # max steering angle (rad)

    c_x = params.get('c_x', 0.045)
    c_y = params.get('c_y', 0.0)

    g = 9.81
    s_width = 0.09 # m
    s_length = 0.09 # m
    s_mass = 1.05 # kg

    f_max = 0.35 * s_mass * g
    slider_area = s_width * s_length
    circular_density_integral = (1 / 12.0) * ((s_length / 2) ** 2 + (s_width / 2) ** 2) * np.exp(1)
    tau_max = 0.35 * s_mass * g * (1 / slider_area) * circular_density_integral

    a = params.get('a', (1 / (f_max ** 2 + (0.35 * f_max) ** 2)))
    b = params.get('b', (1 / (tau_max ** 2)) )

    # Diagonal matrix with magic coeffs
    A = np.diag([a, a, b])

    # Jacobian
    J = np.array([
        [1.0, 0, c_y],
        [0, 1.0, c_x],
    ])

    # Compute rotation matrix
    R = np.array([
        [np.cos(x[2]), -np.sin(x[2]), 0],
        [np.sin(x[2]), np.cos(x[2]), 0],
        [0, 0, 1],
    ])

    # Return the derivative of the state
    return R @ A @ J.T @ u

def vehicle_output(t, x, u, params):
    return x                            # return x, y, theta (full state)

# Constants
data = {'Tf': 12,}

# Define the vehicle steering dynamics as an input/output system
ps1 = ct.NonlinearIOSystem(
    vehicle_update, vehicle_output, states=3, name='pusher-slider1',
    inputs=('f_x', 'f_y'), outputs=('s_x', 's_y', 's_theta'))

x0 = np.array([-0.5, -.5, 0.]); u0 = np.array([10., 0.])
xf = np.array([0.5, 0.5, np.pi/4.0]); uf = np.array([10., 0.])
Tf = data['Tf']

Q = np.diag([0, 0, 0.1])          # don't turn too sharply
R = np.diag([1, 1])               # keep inputs small
P = np.diag([10000, 10000, 10000])   # get close to final point
traj_cost = opt.quadratic_cost(ps1, Q, R, x0=xf, u0=uf)
term_cost = opt.quadratic_cost(ps1, P, 0, x0=xf)

nominal_scenario = {
        "obstacle_center_x": 0.0,
        "obstacle_center_y": 0.0,
        "obstacle_radius": 0.1,
    }
ps2 = PusherSliderStickingForceInputSystem(
    nominal_scenario=nominal_scenario,
)

constraints = [
    opt.input_poly_constraint(ps1, ps2.U.A, ps2.U.b),
    (
        optimize.NonlinearConstraint,
        lambda x, u: np.linalg.norm(x),
        1.5 * nominal_scenario["obstacle_radius"],
        float('Inf'),
    ),
]

timepts = np.linspace(0, Tf, 10, endpoint=True)
result = opt.solve_ocp(
    ps1, timepts, x0, traj_cost, constraints,
    terminal_cost=term_cost, initial_guess=u0)

# Simulate the system dynamics (open loop)
resp = ct.input_output_response(
    ps1, timepts, result.inputs, x0,
    t_eval=np.linspace(0, Tf, 100))
t, y, u = resp.time, resp.outputs, resp.inputs

plt.subplot(3, 1, 1)
plt.plot(y[0], y[1])
plt.plot(x0[0], x0[1], 'ro', xf[0], xf[1], 'ro')
plt.axis([-1.0, 1.0, -1.0, 1.0])
plt.xlabel("x [m]")
plt.ylabel("y [m]")

plt.subplot(3, 1, 2)
plt.plot(t, u[0])
plt.axis([0, 10, -10.0, 10.0])
plt.xlabel("t [sec]")
plt.ylabel("u1 [m/s]")

plt.subplot(3, 1, 3)
plt.plot(t, u[1])
plt.axis([0, 10, -3, 3])
plt.xlabel("t [sec]")
plt.ylabel("u2 [rad/s]")

plt.suptitle("Lane change manuever")
plt.tight_layout()
plt.savefig("data/optimized_traj1.png")

# Compute the Video
# Save results
now = datetime.now()
d4 = now.strftime("%b-%d-%Y-%H:%M:%S")
with open('data/control1-data-' + d4 + '.yml', 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)

# Save the trajectory in a jax form
print(y.shape)
X = jnp.array(y).reshape((1, y.shape[0], y.shape[1]))

U = jnp.array(u).reshape((1, u.shape[0], u.shape[1]))

th = ps2.theta
th = th.reshape((1, th.shape[0]))

ps2.save_animated_trajectory(
    x_trajectory=X,
    th=th,
    f_trajectory=U,
    hide_axes=False,
    filename="test_control_trajopt1.mp4",
    show_obstacle=True, show_goal=True,
)