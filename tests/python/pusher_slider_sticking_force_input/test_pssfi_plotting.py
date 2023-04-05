"""
test_pssfi_plotting.py
Description:
    This script tests a number of the plotting features of the PusherSliderStickingForceInput system.
"""

import unittest
import jax.numpy as jnp
import matplotlib.pyplot as plt

import sys, os
sys.path.append('../../../')
from src.python.pusher_slider_sticking_force_input import PusherSliderStickingForceInputSystem

class TestPSSFIPlotting(unittest.TestCase):
    def test_contact_point1(self):
        # Constants
        scenario = {
            "obstacle_center_x": 0.0,
            "obstacle_center_y": 0.0,
            "obstacle_radius": 0.1,
        }
        ps = PusherSliderStickingForceInputSystem(scenario)
        x = jnp.array([0.0, 0.0, 0.0])

        # Test contact point
        cp = ps.contact_point(x)
        self.assertEqual(cp.at[0].get(), -ps.s_width/2.0)
        self.assertEqual(cp.at[1].get(), 0.0)

    def test_plot_single1(self):
        # Constants
        scenario = {
            "obstacle_center_x": 0.0,
            "obstacle_center_y": 0.0,
            "obstacle_radius": 0.1,
        }
        ps = PusherSliderStickingForceInputSystem(scenario)
        x = jnp.array([-0.3, 0.2, 0.0])
        theta = ps.theta

        # Test plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ps.plot_single(x, theta, ax,
                       limits=[[-0.4, 0.4], [-0.4, 0.4]],
                       show_geometric_center=True,
                       )

        if "/python/pusher_slider_sticking_force_input" in os.getcwd():
            plt.savefig("figures/test_plot_single1.png")

    def test_animate2(self):
        """
        test_animate2
        Description:
            Creates a member function that animates a given state trajectory and
            force trajectory.
        """

        # Constants
        nominal_scenario = {
            "obstacle_center_x": 0.0,
            "obstacle_center_y": 0.0,
            "obstacle_radius": 0.2,
        }
        ps = PusherSliderStickingForceInputSystem(
            nominal_scenario,
        )

        # Create synthetic state, force trajectories
        x0 = jnp.array([[0.1, 0.1, jnp.pi / 6]])
        N_traj = 100
        T_sim = 0.5
        x_trajectory = jnp.array(
            [[t, t, (jnp.pi / 6) * t * 3 * 2] for t in jnp.linspace(0, T_sim, N_traj + 1)]
        ).T
        x_trajectory = x_trajectory.reshape(1, x_trajectory.shape[0], x_trajectory.shape[1])
        th = jnp.array(ps.theta)
        th = th.reshape((1, th.shape[0]))

        f0 = jnp.array([0.1, 0.0])
        f_trajectory = jnp.kron(jnp.ones((N_traj + 1, 1)), f0).T
        f_trajectory = f_trajectory.reshape((1, f_trajectory.shape[0], f_trajectory.shape[1]))

        # Animate with function
        if "/python/pusher_slider_sticking_force_input" in os.getcwd():
            ps.save_animated_trajectory(
                x_trajectory=x_trajectory,
                th=th,
                f_trajectory=f_trajectory,
                hide_axes=False,
                filename="figures/pusherslider-test_animate2.mp4",
                show_obstacle=False, show_goal=False,
            )


if __name__ == '__main__':
    unittest.main()