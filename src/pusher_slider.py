"""
pusher_slider.py
Description:
    Creates a class that represents a Pusher-Slider System.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

class PusherSliderSystem(object):
    """
    __init__
    Description:
        For the sake of cleanliness, I will not include the option to
        initialize parameters with input values. For now, do that outside
        of this initializer.
    """
    def __init__(self):
        # Defaults
        self.s_width = 0.09  # m 
        self.s_length = 0.09 # m
        self.s_mass = 1.05   # kg
        self.ps_cof = 0.3    # Pusher-to-Slider Coefficient of Friction
        self.st_cof = 0.35   # Slider-to-Table Coefficient of Friction
        
        self.p_radius = 0.01 # Radius of the Pusher representation (circle)

        # Define initial state
        self.s_x = 0.1
        self.s_y = 0.1
        self.s_theta = jnp.pi/2
        self.p_x = self.s_width/2
        self.p_y = 0.02

        # Define Initial Input
        self.v_n = 0.01 # Velocity normal to the slider
        self.v_t = 0.03 # Velocity tangential to the slider

    """
    plot
    Description:
        Plots the pusher-slider system onto a new figure.
    """
    def plot(self)->(plt.figure):
        # Constants
        lw = 2.0
        slider_color = 'blue'
        pusher_color = 'magenta'

        # Create Slider
        # =============

        x_lb = -self.s_width/2
        x_ub =  self.s_width/2
        y_lb = -self.s_length/2
        y_ub =  self.s_length/2

        

        




