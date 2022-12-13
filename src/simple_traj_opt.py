"""
simple_traj_opt.py
Description:
    This file defines the functions necessary to run a simple trajectory optimization algorithm on the
    pusher slider system.
"""

from jax import grad
import jax.numpy as jnp
import time
from src.pusher_slider import PusherSliderSystem


"""
simple_endpoint_traj_opt
Description:
    Computes a trajectory that minimizes the distance of the final point to a point x_star.
    The trajectory is composed of N steps each having duration delta_t.
    The optimizer will run N_traj_opt number of gradient descent iterations to reach the target.
Usage:

Variables:
    ps      - An instance of a PusherSliderSystem object.
              The current state of this system is the initial condition of the trajectory optimizer.
    x_star  -
"""


def simple_endpoint_traj_opt(ps:PusherSliderSystem, x_star:jnp.array, N:int=100,N_traj_opt:int=10000,dt:float=0.1,u_step_size=0.01)->tuple[jnp.array,float,float]:
    # Constants
    x0 = ps.x()

    # Define Loss and N-Step Composition Functions
    def NStepCompositionFunction(u: jnp.array):
        # Reshape According to input dimension of ps
        input_dim = 2

        # Compute Composition
        x_t = x0
        for k in range(N):
            u_t = u[k * input_dim:(k + 1) * input_dim, :]
            x_tp1 = x_t + ps.f(x_t, u_t) * dt

            # Set new variable values for next loop
            x_t = x_tp1

        return x_t

    # Create loss
    def loss(u: jnp.array):
        return jnp.linalg.norm(NStepCompositionFunction(u) - x_star)

    # Define Hill Climbing Procedure
    u_init = jnp.zeros((2 * N, 1))

    grad_L = grad(loss)
    u_k = u_init
    opt_start_time = time.time()
    for k in range(N_traj_opt):
        # At each step measure the loss
        print("Loss at", k, "=", loss(u_k))

        # Update input
        u_kp1 = u_k - u_step_size * grad_L(u_k)

        # Set variables for next loop iteration
        u_k = u_kp1

    # Finished with optimization
    opt_end_time = time.time()
    final_loss = loss(u_k)

    return u_k, opt_end_time - opt_start_time, final_loss

"""
ic_traj_opt
Description:
"""
def ic_traj_opt(ps:PusherSliderSystem, x0:jnp.array, x_star:jnp.array, N:int=100,N_traj_opt:int=10000,dt:float=0.1,u_step_size=0.01)->tuple[jnp.array,float,float]:
    # Constants
    ps.set_state(x0)

    # Define Loss and N-Step Composition Functions
    def NStepCompositionFunction(u: jnp.array):
        # Reshape According to input dimension of ps
        input_dim = 2

        # Compute Composition
        x_t = x0
        for k in range(N):
            u_t = u[k * input_dim:(k + 1) * input_dim, :]
            x_tp1 = x_t + ps.f(x_t, u_t) * dt

            # Set new variable values for next loop
            x_t = x_tp1

        return x_t

    # Create loss
    def loss(u: jnp.array):
        return jnp.linalg.norm(NStepCompositionFunction(u) - x_star)

    # Define Hill Climbing Procedure
    u_init = jnp.zeros((2 * N, 1))

    grad_L = grad(loss)
    u_k = u_init
    opt_start_time = time.time()
    for k in range(N_traj_opt):
        # At each step measure the loss
        print("Loss at", k, "=", loss(u_k))

        # Update input
        u_kp1 = u_k - u_step_size * grad_L(u_k)

        # Set variables for next loop iteration
        u_k = u_kp1

    # Finished with optimization
    opt_end_time = time.time()
    final_loss = loss(u_k)

    return u_k, opt_end_time - opt_start_time, final_loss