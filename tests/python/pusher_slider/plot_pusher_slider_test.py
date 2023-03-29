"""
plot_pusher_slider_test.py
Description:

"""

import sys, unittest

import matplotlib.pyplot as plt
import jax.numpy as jnp

sys.path.append('../../../')
from src.python.pusher_slider import PusherSliderSystem

class PusherSliderTest(unittest.TestCase):
    """
    test_plot1
    Description:
        Making sure that this sketch of a plotting algorithm works.
    """
    def test_plot1(self):

        # Constants
        self.s_width = 0.09  # m 
        self.s_length = 0.09 # m

        # Algorithm
        x_lb = -self.s_width/2
        x_ub =  self.s_width/2
        y_lb = -self.s_length/2
        y_ub =  self.s_length/2

        corners = jnp.array( \
            [[x_lb, x_lb, x_ub, x_ub, x_lb],
             [y_lb, y_ub, y_ub, y_lb, y_lb]] \
        )

        # Plotting
        fig1 = plt.figure()
        plt.plot(corners[0,:],corners[1,:])

        fig1.savefig("../../images/testing/pusher_slider_test1-show.png")

    """
    test_plot2
    Description:
        Making sure that this sketch of a plotting algorithm works.
        Creates slider and rotates it Gives the appropriate color.
    """
    def test_plot2(self):

        # Constants
        self.s_width = 0.09  # m 
        self.s_length = 0.09 # m
        self.s_theta = jnp.pi/6 # radians

        self.s_x = 0.1
        self.s_y = 0.1

        lw = 2.0
        slider_color = 'blue'

        # Algorithm
        x_lb = -self.s_width/2
        x_ub =  self.s_width/2
        y_lb = -self.s_length/2
        y_ub =  self.s_length/2

        corners = jnp.array(
            [[x_lb, x_lb, x_ub, x_ub, x_lb],
             [y_lb, y_ub, y_ub, y_lb, y_lb]]
        )

        rot = jnp.array(
            [[jnp.cos(self.s_theta), -jnp.sin(self.s_theta)],
             [jnp.sin(self.s_theta),jnp.cos(self.s_theta)]]
        )

        rotated_corners = rot.dot(corners)

        slider_pos = jnp.array([[self.s_x],[self.s_y]])
        rot_n_transl_corners = rotated_corners + \
            jnp.kron(
                jnp.ones( (1,corners.shape[1]) ) , slider_pos
            )

        # Plotting
        fig1 = plt.figure()
        plt.plot(
            rot_n_transl_corners[0,:],rot_n_transl_corners[1,:],
            linewidth=lw, color=slider_color
        )

        fig1.savefig("../../images/testing/pusher_slider_test2-show.png")

    """
    test_plot3
    Description:
        Making sure that this sketch of a plotting algorithm works.
        - Creates slider
        - Rotates slider
        - Translates slider
        - Color's slider's edge
        - Create pusher
    """
    def test_plot3(self):

        # Constants
        self.s_width = 0.09  # m 
        self.s_length = 0.09 # m
        self.s_theta = jnp.pi/6.0 # radians

        self.s_x = 0.1
        self.s_y = 0.1
        self.p_x = self.s_width/2
        self.p_y = 0.02

        lw = 2.0
        slider_color = 'blue'
        pusher_color = 'magenta'

        self.p_radius = 0.01 # Radius of the Pusher representation (circle)

        # Creating Slider
        # ===============
        x_lb = -self.s_width/2
        x_ub =  self.s_width/2
        y_lb = -self.s_length/2
        y_ub =  self.s_length/2

        corners = jnp.array(
            [[x_lb, x_lb, x_ub, x_ub, x_lb],
             [y_lb, y_ub, y_ub, y_lb, y_lb]]
        )

        rot = jnp.array(
            [[jnp.cos(self.s_theta), -jnp.sin(self.s_theta)],
             [jnp.sin(self.s_theta),jnp.cos(self.s_theta)]]
        )

        rotated_corners = rot.dot(corners)

        slider_pos = jnp.array([[self.s_x],[self.s_y]])
        rot_n_transl_corners = rotated_corners + \
            jnp.kron(
                jnp.ones( (1,corners.shape[1]) ) , slider_pos
            )

        # Plotting
        fig1 = plt.figure()
        ax   = fig1.add_subplot(111)
        plt.plot(
            rot_n_transl_corners[0,:],rot_n_transl_corners[1,:],
            linewidth=lw, color=slider_color
        )

        # Creating Pusher
        # ===============

        # Create circle
        circle_center = jnp.array([[self.s_x],[self.s_y]]) + \
            rot.dot( jnp.array([[-self.p_x],[self.p_y]]) + jnp.array([[-self.p_radius],[0.0]]) )

        ax.add_patch(
            plt.Circle(  (circle_center[0], circle_center[1]) , self.p_radius, color=pusher_color, alpha=0.2)
        )

        fig1.savefig("../../images/testing/pusher_slider_test3-show.png")

    """
    test_plot4
    Description:
       Making sure that the plotting function works.
    """
    def test_plot4(self):

        # Constants
        lw = 2.0
        slider_color = 'blue'
        pusher_color = 'magenta'
        ps1 = PusherSliderSystem()

        # Plotting
        fig1 = plt.figure()
        ps1.plot(fig1)

        fig1.savefig("../../images/testing/pusher_slider_test4-show.png")

if __name__ == '__main__':
    unittest.main()