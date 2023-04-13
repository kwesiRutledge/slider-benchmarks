"""
test_uniform_over_simplex.py
Description:
    In this script, we test jax.scipy minimize's ability to solve the convex combination decomposition problem.
    In Jax!
"""
import jax
import jax.numpy as jnp
import jax.scipy as jsp

from jax import (
    jit, random
)

from functools import partial  # pylint: disable=g-importing-member

from trajax.optimizers import objective

import numpy

import polytope as pc

import matplotlib.pyplot as plt

from optimizers2 import center_of_vertices

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

@partial(jit, static_argnums=(3,4,5))
def gaussian_samples_from_polytope(
        random_key, mean, vertices_U,
        num_samples: int = None,
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
    normalized_ub = jnp.ones((n_vertices,))
    stdev = jnp.array(
        [(normalized_ub - normalized_lb) / 2.] * horizon,
    )

    noises = jax.random.normal(
        random_key, shape=(num_samples, horizon, n_vertices),
    )
    # noises_sums = jnp.sum(noises, axis=-1)

    # Smoothens noise along time axis.
    def body_fun(t, noises):
        return noises.at[:, t].set(smoothing_coef * noises[:, t - 1] +
                               jnp.sqrt(1 - smoothing_coef**2) * noises[:, t])

    noises = jax.lax.fori_loop(1, horizon, body_fun, noises)
    samples_normalized = noises * stdev
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

# Main
# ====

if __name__ == '__main__':
    # Constants
    nominal_scenario = {
        "obstacle_center_x": 0.0,
        "obstacle_center_y": 0.0,
        "obstacle_radius": 0.2,
    }
    ps = PusherSliderStickingForceInputSystem(
        nominal_scenario=nominal_scenario,
    )

    horizon = 10
    n_u = 2

    init_state = jnp.array([-0.5, -0.5, 0.0])

    U0 = jnp.zeros((horizon, n_u))
    U0 = U0.at[:, 0].set(3.0)
    init_controls = U0

    vertices_U = pc.extreme(ps.U)


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


    cost = ps_cost
    dynamics = ps_dynamics

    mean = jnp.array(init_controls)

    print(mean)
    # stdev = np.array([chebR for k in range(H_U.shape[1])] * init_controls.shape[0])

    obj_fn = partial(objective, cost, dynamics)

    # Solve
    res = solve_conversion_opt(
        mean, jnp.array(vertices_U),
    )

    print(res)
    # print(res.x)
    # print(res.success)
    # print(vertices_U)

    # Normalize via mean
    mean_normalized = normalize_via_convex_decomposition(mean, jnp.array(vertices_U))
    # print(mean_normalized)

    # Create fictitious std dev
    normalized_lb = jnp.zeros((n_u,))
    normalized_ub = jnp.ones((n_u,))
    stdev = jnp.array(
        [(normalized_ub - normalized_lb) / 2.] * horizon,
    )

    # Create samples with gaussian_samples_from_polytope
    key0 = random.PRNGKey(0)
    samples = gaussian_samples_from_polytope(
        key0, mean, vertices_U, 5,
    )
    print(
        samples
    )

    # Plot Samples
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ps.U.plot(ax=ax, color='blue',alpha=0.4)
    ax.scatter(
        jnp.asarray(samples)[0, :, 0], jnp.asarray(samples)[0, :, 1],
        # s=1,
    )
    ax.set_xlim(
        -1.0,
        jnp.max(jnp.array([11.0, jnp.max(jnp.asarray(samples)[:, 0])]))
    )
    ax.set_ylim(
        -10.0*ps.st_cof,
        10.0*ps.st_cof,
    )

    fig.savefig('trajax3-samples.png')



