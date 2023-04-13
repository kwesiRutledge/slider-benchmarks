"""
optimizers2.py
Description:
    This script will contain functions that extend trajax's optimizers to work on
    systems with input constraints.
"""

import jax
import jax.numpy as jnp
from jax import (
    jit
)

from functools import partial  # pylint: disable=g-importing-member

import trajax

@partial(jit, static_argnums=(6,7,8))
def gaussian_samples_from_polytope(
        random_key, mean, horizon, dim_control,
        vertices_U,
        num_samples, smoothing_coef,
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
    # Constants
    horizon = mean.shape[0]
    dim_control = mean.shape[1]

    n_vertices = vertices_U.shape[0]

    # Conversion to normalized coordinates
    vectorized_mean = jnp.zeros((mean.shape[0]*mean.shape[1],))
    for k in range(horizon):
        vectorized_mean = vectorized_mean.at[k*dim_control:(k+1)*dim_control].set(
            mean.at[k, :].get()
        )

    kron_VT = jnp.kron(
        jnp.eye(horizon), vertices_U.T,
    )

    # Solve minimization
    obj = lambda theta: jnp.linalg.norm(kron_VT @ theta - vectorized_mean)
    initial_guess = (1.0/n_vertices)*jnp.ones((mean.shape[0]*mean.shape[1],))
    res = jax.scipy.optimize.minimize(
        obj, initial_guess,
    )


    #num_samples = hyperparams['num_samples']
    num_vertices = vertices_U.shape[0]

    noises = jax.random.normal(
        random_key, shape=(num_samples, horizon, num_vertices))
    # Smoothens noise along time axis.
    #smoothing_coef = hyperparams['sampling_smoothing']

    def body_fun(t, noises):
        return noises.at[:, t].set(smoothing_coef * noises[:, t - 1] +
                               np.sqrt(1 - smoothing_coef**2) * noises[:, t])

    noises = jax.lax.fori_loop(1, horizon, body_fun, noises)
    samples = noises * stdev
    samples = samples + mean
    control_low = jax.lax.broadcast(control_low, samples.shape[:-1])
    control_high = jax.lax.broadcast(control_high, samples.shape[:-1])
    samples = np.clip(samples, control_low, control_high)
    return samples

def center_of_vertices(vertices):
  """Computes the center of a set of vertices.

  Args:
    vertices: a numpy array of shape (num_vertices, dim_state)

  Returns:
    center: a numpy array of shape (dim_state,)
  """
  return jnp.mean(vertices, axis=0)

@partial(jit, static_argnums=(0, 1, 9, 11))
def cem_with_input_constraints(cost,
        dynamics,
        init_state,
        init_controls,
        vertices_U,
        random_key=None,
        hyperparams=None,
        max_iter: int =None,
        num_samples: int =None,
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

  mean = center_of_vertices(vertices_U)
  stdev = np.array([chebR for k in range(H_U.shape[1])] * init_controls.shape[0])

  obj_fn = partial(objective, cost, dynamics)

  def loop_body(_, args):
    mean, stdev, random_key = args
    random_key, rng = random.split(random_key)
    controls = gaussian_samples_from_polytope(rng, mean, verices_U,
                                hyperparams['num_samples'], hyperparams['sampling_smoothing'])
    costs = vmap(obj_fn, in_axes=(0, None))(controls, init_state)
    mean, stdev = cem_update_mean_stdev(mean, stdev, controls, costs,
                                        hyperparams['num_samples'],
                                        hyperparams['elite_portion'],
                                        hyperparams['evolution_smoothing'])
    return mean, stdev, random_key

  # TODO(sindhwani): swap with lax.scan to make this optimizer differentiable.
  mean, stdev, random_key = lax.fori_loop(0, hyperparams['max_iter'], loop_body,
                                          (mean, stdev, random_key))

  X = rollout(dynamics, mean, init_state)
  obj = objective(cost, dynamics, mean, init_state)
  return X, mean, obj