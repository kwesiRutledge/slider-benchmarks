"""
test_cem_with_bounded_inputs.py
Description:
    In this script, we test the CEM algorithm with bounded inputs.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from jax import (
    jit, random, lax, vmap,
)

import numpy
from datetime import datetime
import yaml

from functools import partial  # pylint: disable=g-importing-member

from trajax.optimizers import (
    objective, default_cem_hyperparams, cem_update_mean_stdev, rollout,
)

import polytope as pc

import matplotlib.pyplot as plt

import time

import sys
sys.path.append('../../src/python/')
from pusher_slider_sticking_force_input import PusherSliderStickingForceInputSystem

# Functions
# ==========

def solve_conversion_opt(mean: jnp.array, vertices_U: jnp.array):
    """
    res = solve_conversion_opt(mean, vertices_U)
    Description:
        Solve the convex combination decomposition problem. (Decomposes the set of points
        in mean into a set of points in the normalized space according to the vertices of U
        (vertices_U).)
    """
    # Constants
    horizon = mean.shape[0]
    dim_control = mean.shape[1]

    n_vertices = vertices_U.shape[0]

    # Conversion to normalized coordinates
    vectorized_mean = mean.reshape(
        (horizon * dim_control,),
    )

    kron_VT = jnp.kron(
        jnp.eye(horizon), vertices_U.T,
    )

    # Solve minimization
    def obj_fcn(theta):
        temp_obj = jnp.linalg.norm(kron_VT @ theta - vectorized_mean)
        for k in range(horizon):
            temp_obj = temp_obj + jnp.abs(
                jnp.sum(theta[k * n_vertices:(k + 1) * n_vertices]) - 1,
            )
            # Add penalty for negative weights
            for j in range(n_vertices):
                temp_obj = temp_obj + jnp.max(
                    jnp.array([0, -10*theta.at[k * n_vertices + j].get()]),
                )
        return temp_obj

    obj = lambda theta: jnp.linalg.norm(kron_VT @ theta - vectorized_mean) + jnp.linalg.norm(jnp.sum(theta, axis=1) - 1)
    initial_guess = (1.0 / n_vertices) * jnp.ones((horizon*n_vertices,))
    res = jsp.optimize.minimize(
        obj_fcn, initial_guess, method="BFGS",
    )

    return res

def normalize_via_convex_decomposition(mean: jnp.array, vertices_U):
    """
    mean_normalized = normalize_via_convex_decomposition(mean, vertices_U)
    Description:
        Normalize the mean via the convex decomposition.
    """

    # Constants
    horizon = mean.shape[0]
    dim_control = mean.shape[1]

    n_vertices = vertices_U.shape[0]

    # Decompose with optimization
    res = solve_conversion_opt(mean, vertices_U)

    #assert res.success or (res.status == 3), f"Optimization failed: {res.status}"

    mean_normalized = jnp.zeros((horizon, n_vertices))
    for k in range(horizon):
        mean_normalized = mean_normalized.at[k, :].set(
            res.x[k * n_vertices:(k + 1) * n_vertices],
        )

    return mean_normalized

@partial(jit, static_argnums=(4,5,6))
def gaussian_samples_from_polytope(
        random_key, mean, stdev_normalized,
        vertices_U,
        num_samples: int = 10,
        smoothing_coef: float = 0.1,
        max_iter: int = 1000,
):
    """Samples a batch of controls based on Gaussian distribution.

    Args:
    random_key: a jax.random.PRNGKey() random seed
    mean: mean of control sequence, has dimension (horizion, dim_control).
    stdev: stdev of control sequence, has dimension (horizon, dim_control).
    control_low: lower bound of control space.
    control_high: upper bound of control space.
    hyperparams: dictionary of hyperparameters with following keys: num_samples
      -- number of control sequences to sample sampling_smoothing -- a number in
      [0, 1] to control amount of smoothing,
        see eq. 3-4 in https://arxiv.org/pdf/1907.03613.pdf for more details.

    Returns:
    Array of sampled controls, with dimension (num_samples, horizon,
    dim_control).
    """
    # Input Processing
    if random_key is None:
        random_key = random.PRNGKey(0)

    # Constants
    horizon = mean.shape[0]
    dim_control = mean.shape[1]

    n_vertices = vertices_U.shape[0]

    # Conversion to normalized coordinates
    mean_normalized = normalize_via_convex_decomposition(mean, jnp.array(vertices_U))
    normalized_lb = jnp.zeros((n_vertices,))
    # normalized_ub = jnp.ones((n_vertices,))
    # stdev = jnp.array(
    #     [(normalized_ub - normalized_lb) / 2.] * horizon,
    # )

    noises = jax.random.normal(
        random_key, shape=(num_samples, horizon, n_vertices),
    )
    # noises_sums = jnp.sum(noises, axis=-1)

    # Smoothens noise along time axis.
    def body_fun(t, noises):
        return noises.at[:, t].set(smoothing_coef * noises[:, t - 1] +
                               jnp.sqrt(1 - smoothing_coef**2) * noises[:, t])

    noises = jax.lax.fori_loop(1, horizon, body_fun, noises)
    samples_normalized = noises * stdev_normalized
    samples_normalized = samples_normalized + mean_normalized

    control_low = jax.lax.broadcast(normalized_lb, samples_normalized.shape[:-1])
    #control_high = jax.lax.broadcast(normalized_ub, samples_normalized.shape[:-1])
    samples_normalized = jnp.clip(samples_normalized, control_low)

    samples_normalized_sum = jnp.sum(samples_normalized, axis=-1).reshape(
        samples_normalized.shape[0], samples_normalized.shape[1], 1,
    )
    samples_normalized = samples_normalized / samples_normalized_sum

    # Convert Normalized Samples To The Input Space
    samples = convert_normalized_coordinates_to_polytope_samples(
        samples_normalized, vertices_U,
    )
    return samples

def convert_normalized_coordinates_to_polytope_samples(
        samples_normalized: jnp.array, vertices_U: jnp.array,
):
    """
    samples = convert_normalized_coordinates_to_polytope_samples(
        samples_normalized, vertices_U,
    )
    Description:
        Convert normalized coordinates to polytope samples.
    Inputs:
        samples_normalized: samples in normalized coordinates, has dimension
            (num_samples, horizon, n_vertices).
    """

    # Constants
    num_samples = samples_normalized.shape[0]
    horizon = samples_normalized.shape[1]

    n_vertices = vertices_U.shape[0]
    dim_control = vertices_U.shape[1]

    # Algorithm
    samples_normalized_flattened = samples_normalized.reshape(
        (num_samples, horizon * n_vertices, 1)
    )
    kron_VT = jnp.kron(
        jnp.eye(horizon), vertices_U.T,
    )

    # Compute samples
    extended_kron_VT = kron_VT.reshape(
        (1, kron_VT.shape[0], kron_VT.shape[1]),
    )
    extended_kron_VT = jnp.repeat(extended_kron_VT, num_samples, axis=0)

    samples_flattened = extended_kron_VT @ samples_normalized_flattened

    samples = samples_flattened.reshape((num_samples, horizon, dim_control))

    return samples

@partial(jit, static_argnums=(5,6,7))
def cem_update_mean_stdev_normalizedstdev(
        old_mean, old_stdev_normalized, controls, vertices_U,
        costs, num_samples, elite_portion, evolution_smoothing):
  """Computes new mean and standard deviation from elite samples."""
  # Constants
  num_vertices = vertices_U.shape[0]

  #num_samples = hyperparams['num_samples']
  #num_elites = int(num_samples * hyperparams['elite_portion'])

  num_elites = int(num_samples * elite_portion)

  # Finding the elite points
  best_control_idx = jnp.argsort(costs)[:num_elites]
  elite_controls = controls[best_control_idx]
  # Compute New Mean and Standard Deviation
  new_mean = jnp.mean(elite_controls, axis=0)
  elite_controls_normalized = normalize_via_convex_decomposition(
      elite_controls.reshape((elite_controls.shape[0]*elite_controls.shape[1], elite_controls.shape[2])),
      vertices_U,
  ).reshape((elite_controls.shape[0], elite_controls.shape[1], num_vertices))
  new_stdev_normalized = jnp.std(elite_controls_normalized, axis=0)
  # updated_mean = hyperparams['evolution_smoothing'] * old_mean + (
  #     1 - hyperparams['evolution_smoothing']) * new_mean
  # updated_stdev = hyperparams['evolution_smoothing'] * old_stdev + (
  #     1 - hyperparams['evolution_smoothing']) * new_stdev
  updated_mean = evolution_smoothing * old_mean + (
      1 - evolution_smoothing) * new_mean
  updated_stdev = evolution_smoothing * old_stdev_normalized + (
      1 - evolution_smoothing) * new_stdev_normalized

  return updated_mean, updated_stdev

@partial(jit, static_argnums=(0, 1, 7, 8, 9))
def cem_with_input_constraints(cost,
        dynamics,
        init_state,
        init_controls,
        vertices_U,
        random_key=None,
        hyperparams=None,
        max_iter: int= 10,
        num_samples: int= 100,
        elite_portion: float= 0.1,
        ):
  """Cross Entropy Method (CEM).

  CEM is a sampling-based optimization algorithm. At each iteration, CEM samples
  a batch of candidate actions and computes the mean and standard deviation of
  top-performing samples, which are used to sample from in the next iteration.

  Args:
    cost: cost(x, u, t) returns a scalar
    dynamics: dynamics(x, u, t) returns next state
    init_state: initial state
    init_controls: initial controls, of the shape (horizon, dim_control)
    control_low: lower bound of control space
    control_high: upper bound of control space
    random_key: jax.random.PRNGKey() that serves as a random seed
    hyperparams: a dictionary of algorithm hyperparameters with following keys
    sampling_smoothing -- amount of smoothing in action sampling. Refer to
                          eq. 3-4 in https://arxiv.org/pdf/1907.03613.pdf for
                            more details.
    evolution_smoothing -- amount of smoothing in updating mean and standard deviation
    elite_portion -- proportion of samples that is considered elites
    max_iter -- maximum number of iterations
    num_samples -- number of action sequences
                            sampled

  Returns:
    X: Optimal state trajectory.
    U: Optimized control sequence, an array of shape (horizon, dim_control)
    obj: scalar objective achieved.
  """
  if random_key is None:
    random_key = random.PRNGKey(0)
  if hyperparams is None:
    hyperparams = default_cem_hyperparams()

  if max_iter is not None:
    hyperparams['max_iter'] = max_iter

  if num_samples is not None:
    hyperparams['num_samples'] = num_samples

  if elite_portion is not None:
    hyperparams['elite_portion'] = elite_portion


  # Constants
  horizon = init_controls.shape[0]
  n_vertices = vertices_U.shape[0]

  mean = jnp.array(init_controls)
  normalized_lb = jnp.zeros((n_vertices,))
  normalized_ub = jnp.ones((n_vertices,))
  stdev_in_norm_coords = jnp.array([
      [(normalized_ub - normalized_lb) / 2.] * horizon,
  ])

  obj_fn = partial(objective, cost, dynamics)

  def loop_body(_, args):
    mean, stdev_in_norm_coords, random_key = args
    random_key, rng = random.split(random_key)
    controls = gaussian_samples_from_polytope(
        rng, mean, stdev_in_norm_coords, vertices_U,
        num_samples=hyperparams['num_samples'],
        smoothing_coef=hyperparams['sampling_smoothing'])
    costs = vmap(obj_fn, in_axes=(0, None))(controls, init_state)
    mean, stdev_in_norm_coords = cem_update_mean_stdev_normalizedstdev(
        mean, stdev_in_norm_coords, controls, vertices_U, costs,
        hyperparams['num_samples'],
        hyperparams['elite_portion'],
        hyperparams['evolution_smoothing'],
    )
    return mean, stdev_in_norm_coords, random_key

  # TODO(sindhwani): swap with lax.scan to make this optimizer differentiable.
  mean, stdev_in_norm_coords, random_key = lax.fori_loop(0, hyperparams['max_iter'], loop_body,
                                          (mean, stdev_in_norm_coords, random_key))

  X = rollout(dynamics, mean, init_state)
  obj = objective(cost, dynamics, mean, init_state)
  return X, mean, obj

# Main
# ====

if __name__ == '__main__':
    # Constants

    # Define Data
    data = {
        "horizon": 10, "x0": [-0.27, -0.3, jnp.pi/4.], 'dt': 0.1,
        'u0': 3.0, 'n_u': 2,
        'elite_portion': 0.01,
        'max_iter': 60, 'num_samples': 1000,
        'J_u_prefactor_diagonal': [0.005 * (1 / 0.35), 0.0],  # 0.0005,
        'J_obstacle_prefactor': 10.0,
        'J_obstacle_prefactor_decay_rate': 0.1,
    }
    nominal_scenario = {
        "obstacle_center_x": 0.0,
        "obstacle_center_y": 0.0,
        "obstacle_radius": 0.1,
    }
    ps = PusherSliderStickingForceInputSystem(
        nominal_scenario=nominal_scenario,
        dt=data['dt'],
    )

    horizon = data["horizon"]

    x0 = jnp.array(data["x0"])


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
        horizon = data['horizon']

        # Compute Distance to Target
        J_x = jnp.linalg.norm(state[:2] - x_star[:2])

        # Compute Input Cost
        # J_u = (0.5 - (0.45*jnp.exp(time_step)/(jnp.exp(time_step) + jnp.exp(data['horizon']-time_step))))*jnp.linalg.norm(action)
        J_u = action.T @ jnp.diag(jnp.array(data['J_u_prefactor_diagonal'])) @ action
        # J_u = jnp.array(0.0)

        # TODO: Penalize actions outside of motion cone!
        # J_FC = jnp.max(0, jnp.abs(action.at[0].get()) - action.at[1].get()*st_cof)

        # Compute J_obstacle
        J_obstacle = data['J_obstacle_prefactor'] * jnp.abs(
            jnp.linalg.norm(state[:2]) - 1.5 * ps.nominal_scenario['obstacle_radius']
        ) * jnp.exp(jnp.linalg.norm(state[:2]) - 1.5 * ps.nominal_scenario['obstacle_radius'])
        # J_obstacle = \
        #     data['J_obstacle_prefactor'] * \
        #     jnp.exp(- 0.1*(time_step-(horizon/2))) * \
        #     jnp.abs(
        #         jnp.linalg.norm(state[:2]) - 1.5 * ps.nominal_scenario['obstacle_radius']
        #     )

        return J_x.astype(float) + J_u.astype(float) + J_obstacle.astype(float)


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
    U0 = U0.at[:, 0].set(data['u0'])

    hyperparams = default_cem_hyperparams()
    hyperparams['max_iter'] = data['max_iter']
    hyperparams['num_samples'] = data['num_samples']
    hyperparams['elite_portion'] = data['elite_portion']

    trajectory_optimization_start = time.time()
    X, U, opt_obj, = cem_with_input_constraints(
        ps_cost, ps_dynamics,
        x0,
        U0,
        pc.extreme(ps.U),
        max_iter=hyperparams['max_iter'],
        num_samples=hyperparams['num_samples'],
        elite_portion=hyperparams['elite_portion'],
    )
    trajectory_optimization_end = time.time()
    data['trajopt_time1'] = trajectory_optimization_end - trajectory_optimization_start
    data['opt_obj1'] = float(opt_obj)

    data['X'] = numpy.asarray(X)
    data['U'] = numpy.asarray(U)

    # Save results
    now = datetime.now()
    d4 = now.strftime("%b-%d-%Y-%H:%M:%S")
    with open('data/trajax3_data_' + d4 + '.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    # Plot results
    X = X.T
    X = X.reshape((1, X.shape[0], X.shape[1]))
    U = U.T
    U = U.reshape((1, U.shape[0], U.shape[1]))

    print(U)

    th = ps.theta
    th = th.reshape((1, th.shape[0]))

    ps.save_animated_trajectory(
        x_trajectory=X,
        th=th,
        f_trajectory=U,
        hide_axes=False,
        filename="trajax3.mp4",
        show_obstacle=True, show_goal=True,
    )