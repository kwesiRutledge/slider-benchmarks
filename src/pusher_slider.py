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
        self.s_theta = jnp.pi/6.0
        self.p_x = self.s_width/2.0
        self.p_y = 0.02

        # Define Initial Input
        self.v_n = 0.01 # Velocity normal to the slider
        self.v_t = 0.03 # Velocity tangential to the slider

    """
    plot
    Description:
        Plots the pusher-slider system onto a new figure.
    """
    def plot(self,fig_in:plt.figure)->None:
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

        # Create Corners

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

        # Plotting Slider 
        ax   = fig_in.add_subplot(111)
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
        

    """
    get_motion_cone_vectors
    Description:
        Gets the two scalars, gamma_t and gamma_b, which define the motino
        cone vectors.
    Usage:
        gamma_t, gamma_b = ps.get_motion_cone_vectors()
    """ 
    def get_motion_cone_vectors(self)->(float,float):
        # Constants
        g = 10
        f_max = self.st_cof * self.s_mass * g
        m_max = self.st_cof * self.s_mass * g * (self.s_width/2.0)  # The last term is meant to come from
                                                                    # a sort of mass distribution/moment calculation.
        c = f_max / m_max
        mu = self.st_cof # TODO: Which coefficient of friction is this supposed to be?

        gamma_t = ( mu*jnp.power(c,2) - self.p_x * self.p_y + mu * jnp.power(self.p_x,2) )/ \
            ( jnp.power(c,2) + jnp.power(self.p_y,2) - mu * self.p_x*self.p_y )

        gamma_b = ( - mu*jnp.power(c,2) - self.p_x * self.p_y - mu * jnp.power(self.p_x,2) )/ \
            ( jnp.power(c,2) + jnp.power(self.p_y,2) + mu * self.p_x*self.p_y )

        return gamma_t, gamma_b

    """
    identify_mode
    Description:
        Identifies if the input given to the 
    """
    def identify_mode(self,u)->str:
        # Constants
        v_n = u[0]
        v_t = u[1]

        # Algorithm
        gamma_t, gamma_b = self.get_motion_cone_vectors()

        if (v_t <= gamma_t * v_n) and ( v_t >= gamma_b * v_n ):
            return 'Sticking'
        elif v_t > gamma_t * v_n :
            return 'SlidingUp'
        elif v_t < gamma_b * v_n :
            return 'SlidingDown'
        else:
            return 'Error! Unexpected input condition satisfied!'

    