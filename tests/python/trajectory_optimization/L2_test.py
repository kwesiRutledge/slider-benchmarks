"""
L2_tests.py
Description:
    Tests all of the functions related to the pusher slider's simple trajectory optimization tools.
"""

import sys, time
import unittest

from jax import grad
import jax.numpy as jnp

sys.path.append('../../../')
from src.python.pusher_slider import PusherSliderSystem
from src.python.simple_traj_opt import simple_endpoint_traj_opt


class L2TrajectoryOptimizationTest(unittest.TestCase):
    """
    test_get_motion_cone1
    Description:
        Tests that the motion cone computation works as I expect it to.
    """
    def test_calc_gradient(self):

        # Constants
        ps = PusherSliderSystem()
        dt = 0.1
        x0 = ps.x()
        N = 10
        x_star = jnp.array([
            [1.0],[1.0],[jnp.pi/3],[0.0]
        ])

        # Define Function which is the N-th composition of ps.f
        def NStepCompositionFunction(u:jnp.array):
            # Reshape According to input dimension of ps
            input_dim = 2

            # Compute Composition
            x_t = x0
            for k in range(N):
                u_t = u[k*input_dim:(k+1)*input_dim,:]
                x_tp1 = x_t + ps.f(x_t,u_t) * dt

                # Set new variable values for next loop
                x_t = x_tp1

            return x_t

        # Create Loss
        u_test = jnp.zeros( (2*N,1) )
        Loss_test = jnp.linalg.norm( NStepCompositionFunction(u_test) - x_star)
        self.assertEqual( Loss_test, jnp.linalg.norm(x0 - x_star) )

        # Compute Gradient
        def Loss_test(u:jnp.array):
            return jnp.linalg.norm( NStepCompositionFunction(u) - x_star )

        grad_Lt = grad(Loss_test)
        print(grad_Lt(u_test))

    """
    test_hill_climb
    Description:
        The goal is to use this function to do trajectory optimization using a simple hill climbing approach.
    """
    def test_hill_climb(self):

        # Constants
        ps = PusherSliderSystem()
        dt = 0.1
        x0 = ps.x()
        N = 100
        x_star = jnp.array([
            [1.0], [1.0], [jnp.pi / 3], [0.0]
        ])

        # Define Function which is the N-th composition of ps.f
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
        def Loss(u: jnp.array):
            return jnp.linalg.norm(NStepCompositionFunction(u) - x_star)

        # Define Hill Climbing Procedure
        u_step_size = 0.1
        u_init = jnp.zeros( (2*N,1) )
        L0 = Loss(u_init)

        grad_L = grad(Loss)
        u_k = u_init
        N_traj_opt = 10
        startOpt = time.time()
        for k in range(N_traj_opt):
            # At each step measure the loss
            print("Loss at", k, "=", Loss(u_k))

            # Update input
            u_kp1 = u_k - u_step_size*grad_L(u_k)

            # Set variables for next loop iteration
            u_k = u_kp1

        endOpt = time.time()

        print("Final Loss =", Loss(u_k))
        print("Time taken = ", endOpt- startOpt)

    """
    test_hill_climb2
    Description:
        The goal is to use this function to do trajectory optimization using a simple hill climbing approach.
    """
    def test_hill_climb2(self):

        # Constants
        ps = PusherSliderSystem()
        x_target = jnp.array([
            [1.0], [1.0], [jnp.pi / 3], [0.0]
        ])

        u_opt, opt_time, final_loss = simple_endpoint_traj_opt(
            ps, x_target,
            N=100, N_traj_opt=10, dt=0.1, u_step_size=0.1
        )
        self.assertLess(final_loss,1.0) # We know that the final loss should be less than this value because of the
                                        # previous test.


if __name__ == '__main__':
    unittest.main()