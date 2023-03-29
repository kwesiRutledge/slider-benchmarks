"""
dynamics_tests.py
Description:
    Tests all of the functions related to the pusher slider's dynamics.
"""

import sys, unittest

import matplotlib.pyplot as plt
import jax.numpy as jnp

sys.path.append('../../../')
from src.python.pusher_slider import PusherSliderSystem

class PusherSlider_CircleTest1(unittest.TestCase):
    """
    test_hogan_rodriguez_dynamics1
    Description:
    	Simulates the f() function so that we know that the state of the pusher slider is not modified
		after evaluating f at a given point.
    """   
    def test_hogan_rodriguez_dynamics1(self):
        # Constants
        ps = PusherSliderSystem()

        x0 = ps.x()
        x0_prime = x0
        x0_prime = x0_prime.at[3,0].set(0.99*ps.s_length)

        dim_x = 4
        T = 15.0
        dt = 0.05

        data_dir_name = "../../../data/modeling1/"
        image_filename1 = data_dir_name + "hogan_circle.png"
        image_filename2 = data_dir_name + "hogan_circle_all_dims.png"

        # Simulate with constant input
        u0 = jnp.array([[0.1], [0.0]])
        num_steps = int(T / dt)
        # print(num_steps)
        xs = jnp.zeros((num_steps+1,dim_x))
        print(x0)
        xs = xs.at[0,:].set(x0_prime.flatten())

        for k in range(1, num_steps+1):
            print(xs[k-1, :])
            x_k = jnp.reshape(xs[k-1, :], (dim_x, 1))
            x_dot = ps.f(x_k, u0)
            # print(x_dot)

            x_next = x_k + dt * x_dot
            xs = xs.at[k,:].set(x_next.flatten())

        print(xs)

        fig1 = plt.figure()
        plt.plot(xs[:, 0],xs[:, 1])
        fig1.savefig(image_filename1)

        nrows, ncols = 4, 1
        fig2 = plt.figure()
        for dim_index in range(4):
            ax_i = plt.subplot(nrows, ncols, dim_index+1)
            plt.plot(jnp.arange(0.0, T+dt, dt), xs[:, dim_index])
        fig2.savefig(image_filename2)

        # Save data to a file that matlab can read.


if __name__ == '__main__':
    unittest.main()