"""
pusher_slider_sticking_force_input.py
Description:
    This file contains functions relevant to performing jax operations on.
"""

from typing import Callable, Tuple, Optional, List, Dict
import jax.numpy as jnp

# Define Scenario
Scenario = Dict[str, float]

class AdaptivePusherSliderStickingForceInputSystem(object):
    """
    PusherSliderStickingForceInputSystem
    Description:
        This class is meant to represent the PusherSlider system with sticking contact
        and force input.
    """

    # Number of states, controls and paramters
    N_DIMS = 3
    N_CONTROLS = 2
    N_PARAMETERS = 2

    # State Indices
    S_X = 0
    S_Y = 1
    S_THETA = 2

    # Control indices
    F_X = 0
    F_Y = 1

    # Parameter indices
    C_X = 0
    C_Y = 1

    def __init__(
            self,
            s_mass=1.05,
            s_width=0.09,
            ps_cof: float = 0.30):
        """
        __init__
        Description:
            Construction of the PusherSlider system with sticking input.
        """

        self.s_width = s_width
        self.s_length = s_width
        self.s_mass = s_mass

        self.ps_cof = ps_cof  # Pusher-to-Slider Coefficient of Friction
        self.st_cof = 0.35  # Slider-to-Table Coefficient of Friction

        self.p_radius = 0.01  # Radius of the Pusher representation (circle)

        self.st_cof = 0.35

    def friction_cone_extremes(self) -> (jnp.array, jnp.array):
        """
        [f_l, f_u] = self.friction_cone_extremes()

        Description:
            This function returns two unit vectors defining the boundary of the friction cone.
            The friction cone vectors are written in the frame of reference at the contact point
            with:
            - positive x being along the edge of the slider and
            - positive y being perpendicular and into the slider.
        Args:

        Outputs:
            f_u: A vector in the direction of the "upper" edge of the friction cone
            f_l: A vector in the direction of the "lower" edge of the friction cone
        """

        # Constants
        mu = self.ps_cof

        # Create output
        return jnp.array([mu, 1.0]), jnp.array([-mu, 1.0])

    @property
    def n_dims(self) -> int:
        return AdaptivePusherSliderStickingForceInputSystem.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return [AdaptivePusherSliderStickingForceInputSystem.S_THETA]

    @property
    def parameter_angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return AdaptivePusherSliderStickingForceInputSystem.N_CONTROLS

    @property
    def n_params(self) -> int:
        return AdaptivePusherSliderStickingForceInputSystem.N_PARAMETERS

    def goal_state(self, theta: jnp.array) -> jnp.array:
        """
        goal_point
        Description:
            In this case, we force the goal state to be the same point [0.5,0.5,0]
            for any theta input.
        """
        # Defaults
        if theta is None:
            theta = jnp.zeros(1, self.n_params)

        # Constants
        batch_size = theta.shape[0]

        # Algorithm
        goal = jnp.ones((batch_size, self.n_dims))
        goal *= 0.5  # goal is in the upper right corner of the workspace
        goal = goal.at[:, AdaptivePusherSliderStickingForceInputSystem.S_THETA].set(0.0)

        return goal

    def limit_surface_bounds(self):
        # Constants
        g = 9.8

        # Create output
        f_max = self.st_cof * self.s_mass * g

        slider_area = self.s_width * self.s_length
        # circular_density_integral = 2*pi*((ps.s_length/2)^2)*(1/2)
        circular_density_integral = (1 / 12) * ((self.s_length / 2) ** 2 + (self.s_width / 2) ** 2) * jnp.exp(1)

        tau_max = self.st_cof * self.s_mass * g * (1 / slider_area) * circular_density_integral
        return f_max, tau_max

    def _f(self, x: jnp.array, params: Scenario) -> jnp.array:
        """
        Return the control-independent part of the control-affine dynamics.
        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        # Constants
        batch_size = x.shape[0]
        f = jnp.zeros((batch_size, self.n_dims, 1))

        return f

    def _F(self, x: jnp.array, params: Scenario) -> jnp.array:
        """
        Return the control-independent part of the control-affine dynamics.
        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            F: bs x self.n_dims x self.n_params tensor
        """
        # Constants
        batch_size = x.shape[0]
        F = jnp.zeros((batch_size, self.n_dims, self.n_params))

        return F

    def _g(self, x: jnp.array, params: Scenario) -> jnp.array:
        """
        Return the control-dependent part of the control-affine dynamics.
        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            g: bs x self.n_dims x self.n_controls tensor
        """
        # Constants
        batch_size = x.shape[0]
        g = jnp.zeros((batch_size, self.n_dims, self.n_controls))

        f_max, tau_max = self.limit_surface_bounds()
        a = (1 / 10.0) * (1 / (f_max ** 2))
        b = (1 / 10.0) * (1 / (tau_max ** 2))

        # States
        s_x = x.at[:, AdaptivePusherSliderStickingForceInputSystem.S_X].get()
        s_y = x.at[:, AdaptivePusherSliderStickingForceInputSystem.S_Y].get()
        s_theta = x.at[:, AdaptivePusherSliderStickingForceInputSystem.S_THETA].get()

        # Algorithm
        g = g.at[:, AdaptivePusherSliderStickingForceInputSystem.S_X, AdaptivePusherSliderStickingForceInputSystem.F_X].set(jnp.cos(
            s_theta) * a)
        g = g.at[:, AdaptivePusherSliderStickingForceInputSystem.S_X, AdaptivePusherSliderStickingForceInputSystem.F_Y].set(-jnp.sin(
            s_theta) * a)

        g = g.at[:, AdaptivePusherSliderStickingForceInputSystem.S_Y, AdaptivePusherSliderStickingForceInputSystem.F_X].set(jnp.sin(
            s_theta) * a)
        g = g.at[:, AdaptivePusherSliderStickingForceInputSystem.S_Y, AdaptivePusherSliderStickingForceInputSystem.F_Y].set(jnp.cos(
            s_theta) * a)

        return g

    def _G(self, x: jnp.array, params: Scenario) -> jnp.array:
        """
        Return the control-dependent and parameter-dependent part of the control-affine dynamics.
        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            G: bs x self.n_dims x self.n_controls x self.n_params tensor
        """
        # Constants
        batch_size = x.shape[0]

        f_max, tau_max = self.limit_surface_bounds()
        a = (1 / 10.0) * (1 / (f_max ** 2))
        b = (1 / 10.0) * (1 / (tau_max ** 2))

        # States
        s_x = x[:, AdaptivePusherSliderStickingForceInputSystem.S_X]
        s_y = x[:, AdaptivePusherSliderStickingForceInputSystem.S_Y]
        s_theta = x[:, AdaptivePusherSliderStickingForceInputSystem.S_THETA]

        # Create output
        G = torch.zeros((batch_size, self.n_dims, self.n_controls, self.n_params)).to(self.device)

        G[:, AdaptivePusherSliderStickingForceInputSystem.S_THETA, AdaptivePusherSliderStickingForceInputSystem.F_X,
        AdaptivePusherSliderStickingForceInputSystem.C_Y] = -b
        G[:, AdaptivePusherSliderStickingForceInputSystem.S_THETA, AdaptivePusherSliderStickingForceInputSystem.F_Y,
        AdaptivePusherSliderStickingForceInputSystem.C_X] = b

        return G

