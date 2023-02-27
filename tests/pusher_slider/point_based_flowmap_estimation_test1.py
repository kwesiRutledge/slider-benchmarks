"""
point_based_flowmap_estimation_test1.py
Description:
    Consider the flowmap \phi(x,u,t) which defines the state that the system will be if it starts at state x
    uses the constant input u for t time units.
"""

import sys, unittest

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy, datetime
import scipy.io

sys.path.append('../../')
from src.pusher_slider import PusherSliderSystem

class PusherSlider_CircleTest1(unittest.TestCase):
    """
    test_single_point_dataset1
    Description:
    	Collects a single dataset D(x) for a given state of the system to approximate its flowmap.
    """
    def test_single_point_dataset1(self):
        # Constants
        ps = PusherSliderSystem()

        x0 = ps.x()
        x0_prime = x0
        x0_prime = x0_prime.at[3,0].set(0.99*ps.s_length)

        dim_x = 4
        dim_u = 2
        T = 15.0
        dt = 0.05

        data_dir_name = "../../data/flowmap1/"
        image_filename1 = data_dir_name + "dataset_trajectories.png"
        image_filename2 = data_dir_name + "hogan_circle_all_dims.png"
        data_filename1 = data_dir_name + "dataset_x_" + datetime.datetime.now().strftime("%B%d%Y-%I%M%p") + ".mat"

        # Create a set of control inputs
        samples_per_U_dim = 10
        nU = samples_per_U_dim ** 2
        vn_min, vn_max, vt_min, vt_max = 0.0, 1.0, -0.5, 0.5
        U_cand = jnp.zeros((dim_u, nU))
        count = 0
        for vt in jnp.linspace(vn_min, vn_max, samples_per_U_dim):
            for vn in jnp.linspace(vt_min, vt_max, samples_per_U_dim):
                U_cand = U_cand.at[:, count].set(
                    jnp.array([[vt], [vn]]).flatten()
                )
                count += 1

        print("Candidate u's:")
        print(U_cand)

        # Create dataset by iterating through each candidate input.
        D_x = []

        t_sim = 0.7
        sim_steps = 10
        dt = t_sim/sim_steps
        for u_index in range(U_cand.shape[1]):
            # Extract input
            u_i = U_cand.at[:, u_index].get()
            u_i = jnp.reshape(u_i, (dim_u, 1))

            # Simulate trajectory using inexact method
            xs = jnp.zeros((sim_steps + 1, dim_x))
            xs = xs.at[0, :].set(x0.flatten())

            x_k = x0
            for sim_step in range(sim_steps):
                x_kp1 = x_k + ps.f(x_k, u_i)*dt
                xs = xs.at[sim_step+1, :].set(x_kp1.flatten())

                # Update x_k
                x_k = x_kp1

            D_x.append(
                {"x": x0, "u": u_i, "phi(x,u,tau)": x_k, "traj": xs}
            )

        # Plot each trajectory from this dataset
        fig1 = plt.figure()
        for datapoint in D_x:
            xs = datapoint["traj"]
            plt.plot(
                xs.at[:, 0].get(),
                xs.at[:, 1].get(),
            )

        plt.title("Dataset Trajectories")
        fig1.savefig(image_filename1)

        # Save Data to File that can be read by matlab
        D_x_matlab = []
        for d in D_x:
            d_matlab = {}
            for key1 in d:
                d_matlab[key1] = numpy.array(d[key1])

            # Append this data file to D_x
            D_x_matlab.append(
                d_matlab
            )

        # Save data to file
        scipy.io.savemat(data_filename1, {'dataset': D_x_matlab})






if __name__ == '__main__':
    unittest.main()