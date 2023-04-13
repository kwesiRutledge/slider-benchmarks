"""
traj_opt_scalar.py
Description:
    Trajectory optimization for the scalar system in the safety case study.
"""

import trajax
import trajax.optimizers as optimizers

import numpy
import jax.numpy as jnp

import time

import yaml
from datetime import datetime

# Helper Functions

if __name__ == '__main__':
    # Constants
    data = {
        'horizon': 20, 'n_u': 1, 'u0': -0.5, 'u_max': 10.0,
        'max_iter': 300, 'num_samples': 5000, 'elite_portion': 0.01,
    }
    x0s = jnp.arange(-0.3, 2.1, 0.2)
    x0s = x0s.reshape((x0s.shape[0], 1))

    # Get System
    theta = jnp.array([-2.0])

    # Define Cost and Dynamics Model
    def scalar_cost(x: jnp.array, u: jnp.array, t:float)->float:
        """
        scalar_cost
        Description:
            This function defines the cost of the scalar system.
            We mainly want to avoid the wall at -0.5 and reach the target area at
            1.0.
        """
        # Constants
        wall_pos = -0.5
        target_pos = 1.0

        # Costs
        J_x = jnp.linalg.norm(x - target_pos)

        J_u = jnp.linalg.norm(u) # Fuel Cost

        J_obstacle = wall_pos - x # Avoid Wall

        return J_x.astype(float) + J_u.astype(float) + J_obstacle.astype(float)

    def scalar_dynamics(x: jnp.array, u: jnp.array, t:float)->jnp.array:
        """
        scalar_dynamics
        Description:
            This function defines the dynamics of the scalar system.
        """
        xdot = (1.0 + theta[0])*x + (1.0 + theta[0])*u
        return xdot


    # Perform Trajectory Optimization on each initial condition
    for x0_index in range(x0s.shape[0]):
        x0 = x0s.at[x0_index, :].get()

        # Run cem trajopt
        U0 = jnp.zeros((data['horizon'], data['n_u']))
        U0 = U0.at[:, 0].set(data['u0'])

        hyperparams = optimizers.default_cem_hyperparams()
        hyperparams['max_iter'] = data['max_iter']
        hyperparams['num_samples'] = data['num_samples']

        trajectory_optimization_start = time.time()
        X, U, opt_obj, = optimizers.cem(
            scalar_cost, scalar_dynamics,
            x0,
            U0,
            jnp.array([-data['u_max']]),
            jnp.array([data['u_max']]),
            max_iter=hyperparams['max_iter'],
            num_samples=hyperparams['num_samples'],
            elite_portion=hyperparams['elite_portion'],
        )
        trajectory_optimization_end = time.time()
        data['trajopt_time' + str(x0_index)] = trajectory_optimization_end - trajectory_optimization_start
        data['opt_obj' + str(x0_index)] = float(opt_obj)
        data['U' + str(x0_index)] = numpy.array(U).tolist()
        data['X' + str(x0_index)] = numpy.array(X).tolist()

    # Save number of ics there are
    data['x0s'] = numpy.array(x0s).tolist()
    data['num_x0s'] = x0s.shape[0]

    # Save results
    now = datetime.now()
    d4 = now.strftime("%b-%d-%Y-%H:%M:%S")
    with open('data/safety_scalar_case_study_data_' + d4 + '.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)