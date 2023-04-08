"""
pusher_slider_sticking_force_input.py
Description:
    This file contains functions relevant to performing jax operations on.
"""

from typing import Callable, Tuple, Optional, List, Dict
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

import numpy

import time
import polytope as pc

# Define Scenario
Scenario = Dict[str, float]

class PusherSliderStickingForceInputSystem(object):
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
            nominal_scenario: Scenario,
            s_mass=1.05,
            s_width=0.09,
            ps_cof: float = 0.30,
            dt: float = 0.01,
            max_force: float = 10.0,
    ):
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

        self.theta = jnp.array([0.0, s_width/2.0])

        self.dt = dt
        self.max_force = max_force

        # Save scenario
        assert self.validate_scenario(nominal_scenario)
        self.nominal_scenario = nominal_scenario

    def validate_scenario(self, s: Scenario) -> bool:
        """
        tf = self.validate_scenario(s)

        Description:
            Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
        returns:
            True if parameters are valid, False otherwise
        """
        valid = True

        valid = valid and ("obstacle_center_x" in s)
        valid = valid and ("obstacle_center_y" in s)
        valid = valid and ("obstacle_radius" in s)

        return valid

    @property
    def U(self) -> pc.Polytope:
        """
        P_U = self.U

        Description:
            Return the polytope representing the control constraints for the given system.
        """
        # Define the matrices which define the polytope
        H = numpy.array([
            [1.0, -self.ps_cof],
            [-1.0, -self.ps_cof],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ])

        h = numpy.zeros((5, 1))
        h[2, 0] = self.max_force
        h[3, 0] = self.max_force

        return pc.Polytope(H, h)

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
        return PusherSliderStickingForceInputSystem.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return [PusherSliderStickingForceInputSystem.S_THETA]

    @property
    def parameter_angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return PusherSliderStickingForceInputSystem.N_CONTROLS

    @property
    def n_params(self) -> int:
        return PusherSliderStickingForceInputSystem.N_PARAMETERS

    def goal_state(self, theta: jnp.array, s: Scenario = None) -> jnp.array:
        """
        goal_point
        Description:
            In this case, we force the goal state to be the same point [0.5,0.5,0]
            for any theta input.
        """
        # Defaults
        if theta is None:
            theta = jnp.zeros(1, self.n_params)

        if s is None:
            s = self.nominal_scenario

        # Constants

        # Algorithm
        return jnp.array([0.5, 0.5, 0.0])

    def limit_surface_bounds(self):
        # Constants
        g = 9.8

        # Create output
        f_max = self.st_cof * self.s_mass * g

        slider_area = self.s_width * self.s_length
        # circular_density_integral = 2*pi*((ps.s_length/2)^2)*(1/2)
        circular_density_integral = (1 / 12.0) * ((self.s_length / 2) ** 2 + (self.s_width / 2) ** 2) * jnp.exp(1)

        tau_max = self.st_cof * self.s_mass * g * (1 / slider_area) * circular_density_integral
        return f_max, tau_max

    def compute_motion_cone_factors(self):
        """
        a, b = self.compute_motion_cone_factors()

        """
        # Constants

        # Create output
        f_max, tau_max = self.limit_surface_bounds()

        # a = (1 / (f_max ** 2))
        # b = (1 / (tau_max ** 2))

        # a = 1.0537  # Copied from paper
        # b = 1.5087  # Copied from paper

        # a = (1 / jnp.sqrt(f_max ** 2 + self.ps_cof * f_max ** 2))
        # b = (1 / (tau_max))

        a = (1 / (f_max ** 2 + (self.ps_cof * f_max) ** 2))
        b = (1 / (tau_max ** 2))

        return a, b

    def _f(self, x: jnp.array, s: Scenario = None) -> jnp.array:
        """
        Return the control-independent part of the control-affine dynamics.
        args:
            x: self.n_dims array of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: self.n_dims jax array
        """
        # Input Processing
        assert x.shape[0] == self.n_dims
        assert x.shape == (self.n_dims,)

        if s is None:
            s = self.nominal_scenario

        # Constants
        f = jnp.zeros((self.n_dims,))

        return f

    def _F(self, x: jnp.array, s: Scenario = None) -> jnp.array:
        """
        Return the control-independent part of the control-affine dynamics.
        args:
            x: self.n_dims array of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            F: self.n_dims x self.n_params array
        """
        # Input Processing
        assert x.shape[0] == self.n_dims
        assert x.shape == (self.n_dims,)

        if s is None:
            s = self.nominal_scenario

        # Constants
        F = jnp.zeros((self.n_dims, self.n_params))

        return F

    def _g(self, x: jnp.array, s: Scenario = None) -> jnp.array:
        """
        Return the control-dependent part of the control-affine dynamics.
        args:
            x: self.n_dims array of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            g: self.n_dims x self.n_controls array
        """
        # Input Processing
        assert x.shape[0] == self.n_dims
        assert x.shape == (self.n_dims,)

        if s is None:
            s = self.nominal_scenario

        # Constants
        g = jnp.zeros((self.n_dims, self.n_controls))

        a, b = self.compute_motion_cone_factors()

        # States
        s_x = x.at[PusherSliderStickingForceInputSystem.S_X].get()
        s_y = x.at[PusherSliderStickingForceInputSystem.S_Y].get()
        s_theta = x.at[PusherSliderStickingForceInputSystem.S_THETA].get()

        # Algorithm
        g = g.at[PusherSliderStickingForceInputSystem.S_X, PusherSliderStickingForceInputSystem.F_X].set(
            jnp.cos(s_theta) * a,
        )
        g = g.at[PusherSliderStickingForceInputSystem.S_X, PusherSliderStickingForceInputSystem.F_Y].set(
            -jnp.sin(s_theta) * a,
        )

        g = g.at[PusherSliderStickingForceInputSystem.S_Y, PusherSliderStickingForceInputSystem.F_X].set(
            jnp.sin(s_theta) * a,
        )
        g = g.at[PusherSliderStickingForceInputSystem.S_Y, PusherSliderStickingForceInputSystem.F_Y].set(
            jnp.cos(s_theta) * a,
        )

        return g

    def _G(self, x: jnp.array, s: Scenario = None) -> jnp.array:
        """
        G = self._G(x, params)
        Return the control-dependent and parameter-dependent part of the control-affine dynamics.
        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            G: bs x self.n_dims x self.n_controls x self.n_params tensor
        """
        # Input Processing
        assert x.shape[0] == self.n_dims
        assert x.shape == (self.n_dims,)

        if s is None:
            s = self.nominal_scenario

        # Constants
        a, b = self.compute_motion_cone_factors()

        # States
        s_x = x.at[PusherSliderStickingForceInputSystem.S_X].get()
        s_y = x.at[PusherSliderStickingForceInputSystem.S_Y].get()
        s_theta = x.at[PusherSliderStickingForceInputSystem.S_THETA].get()

        # Create output
        G = jnp.zeros((self.n_dims, self.n_controls, self.n_params))

        G = G.at[PusherSliderStickingForceInputSystem.S_THETA, PusherSliderStickingForceInputSystem.F_X,
            PusherSliderStickingForceInputSystem.C_Y].set(-b)
        #print("G = ", G)

        G = G.at[PusherSliderStickingForceInputSystem.S_THETA, PusherSliderStickingForceInputSystem.F_Y,
            PusherSliderStickingForceInputSystem.C_X].set(b)

        # print("G = ", G)

        return G

    def input_gain_matrix(self, x: jnp.array, theta: jnp.array = None, s: Scenario = None) -> jnp.array:
        """
        Return the factor that multiplies the control value in the dynamics.
        args:
            x: self.n_Dims tensor of state
            theta: self.n_params tensor of state
            params: a dictionary giving the parameter values for the system.
                    If None, default to the nominal parameters used at initialization.
        returns
            g_like: self.n_dims x self.n_controls tensor defining how input vector impacts the state
                    in each batch.
        """
        # Input Processing
        assert x.shape == (self.n_dims,)
        assert theta.shape == (self.n_params,)

        if s is None:
            s = self.nominal_scenario

        if theta is None:
            theta = self.theta

        # Constants

        # Algorithm
        g_like = self._g(x)
        G = self._G(x)
        for param_index in range(self.n_params):
            theta_i = theta.at[param_index].get()
            G_i = jnp.zeros((self.n_dims, self.n_controls))
            G_i = G_i.at[:, :].set(G.at[:, :, param_index].get())
            # Update g
            g_like = g_like + theta_i * G_i

        return g_like

    def control_affine_dynamics(
            self, x: jnp.array, s: Scenario = None
    ) -> Tuple[jnp.array, jnp.array]:
        """
        f, g = self.control_affine_dynamics(x, theta, params)

        description:
            Return a tuple (f + F \theta, g + \sum_i  G_i \theta) representing the system dynamics in control-affine form:
                dx/dt = f(x) + F(x) \theta + { g(x) + \sum_i G(x) \theta_i } u
        args:
            x: self.n_dims tensor of state
            theta: self.n_params tensor of parameter data
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: self.n_dims tensor representing the control-independent dynamics
            g: self.n_dims x self.n_controls tensor representing the control-
               dependent dynamics
        """
        # Input Processing
        theta = self.theta

        assert x.shape == (self.n_dims,)
        assert self.theta.shape == (self.n_params,)

        if s is None:
            s = self.nominal_scenario

        # # If no params required, use nominal params
        # if params is None:
        #     params = self.nominal_scenario

        # print("self._f(x)", self._f(x))
        # print("self._F(x) @ theta", self._F(x) @ theta)
        # print("self.input_gain_matrix(x, theta)", self.input_gain_matrix(x, theta))

        return self._f(x) + self._F(x) @ theta, self.input_gain_matrix(x, theta)

    def closed_loop_dynamics(
            self, x: jnp.array, u: jnp.array, s: Optional[Scenario] = None
    ) -> jnp.array:
        """
        Return the state derivatives at state x and control input u
            dx/dt = f(x) + F(x) \theta + { g(x) + sum_i G(x) \theta_i } u
        args:
            x: bs x self.n_dims tensor of state
            u: bs x self.n_controls tensor of controls
            theta: bs x self.n_params tensor of parameters
            scenario: a dictionary giving the scenario parameter values for the system.
                        If None, default to the nominal parameters used at initialization
        returns:
            xdot: bs x self.n_dims tensor of time derivatives of x
        """
        # Input Processing
        theta = self.theta

        assert x.shape == (self.n_dims,), f"Expected x to be of shape ({self.n_dims},); received shape {x.shape}"
        assert theta.shape == (self.n_params,), f"Expected theta to be of shape ({self.n_params},); received shape {theta.shape}"

        if s is None:
            s = self.nominal_scenario

        # Get the control-affine dynamics
        f, g = self.control_affine_dynamics(x)

        # Compute state derivatives using control-affine form
        xdot = f + g @ u

        return xdot

    def contact_point(self, x: jnp.array) -> jnp.array:
        """
        contact_point
        Description:
            This function returns the contact point of the slider with the pusher.

        Returns
            contact_point: length 2 array, representing of the contact point in the world frame
        """

        # Input Processing
        assert x.shape == (3,), f"Expected x to be of shape (3,); received shape {x.shape}"

        # Constants

        # Get State
        s_x = x.at[0].get()
        s_y = x.at[1].get()
        s_th = x.at[2].get()

        # Compute contact point
        rot2 = jnp.array([[jnp.cos(s_th), -jnp.sin(s_th)], [jnp.sin(s_th), jnp.cos(s_th)]])
        contact_point = jnp.array([[s_x], [s_y]]) + rot2 @ jnp.array([[-self.s_length / 2], [0]])

        return contact_point.reshape((2,))

    def goal_point(self, theta: jnp.array) -> jnp.array:
        """
        goal_point
        Description:
            In this case, we force the goal state to be the same point [0.5,0.5,0]
            for any theta input.
        """
        # Defaults
        if theta is None:
            theta = jnp.zeros((3,))

        # Constants

        # Algorithm

        return jnp.array([0.5, 0.5, 0.0])

    @property
    def goal_radius(self):
        """Return the radius of the goal region"""
        return 0.1

    def plot_single(self,
                    x: jnp.array, theta: jnp.array,
                    ax: plt.Axes,
                    limits: Optional[List[List[float]]] = [[0.0, 0.3], [0.0, 0.3]],
                    hide_axes: bool = True,
                    equal_aspect: bool = True,
                    show_geometric_center: bool = False,
                    show_CoM: bool = True,
                    show_friction_cone_vectors: bool = True,
                    show_obstacle: bool = True,
                    show_goal: bool = True,
                    current_force: jnp.array = None) -> plt.Figure:
        """
        plot_single
        Description:
            Plots a single pusher-slider system in a 2d plot and returns the figure.
        """

        # Input Checking
        assert (x.shape == (3,)), f"x must have shape ({self.n_dims},); received {x.shape}."
        assert (theta.shape == (2,)), f"theta must have shape ({self.n_params},); received {theta.shape}."

        # We can only visualize one state at a time.

        # Constants
        s_length = self.s_length
        s_width = self.s_width
        p_radius = self.p_radius

        # Get state
        s_x = x.at[0].get()
        s_y = x.at[1].get()
        s_th = x.at[2].get()

        # Get parameters
        CoM_x = theta.at[0].get()
        CoM_y = theta.at[1].get()

        # Setup Figure
        # =========
        fig = plt.gcf()
        # ax = fig.add_subplot(111)
        if hide_axes:
            ax.axis('off')

        # Set Axes Limits
        plt.xlim(limits[0])
        plt.ylim(limits[1])

        if equal_aspect:
            plt.gca().set_aspect('equal', adjustable='box')

        plot_objects = {}

        # Plot Objects
        # ============

        # Plot Obstacle first (just in case we collide with it, we want to make it clear that collision happens)
        if show_obstacle:
            s = self.nominal_scenario
            obstacle_x, obstacle_y, obstacle_radius = s["obstacle_center_x"], s["obstacle_center_y"], s[
                "obstacle_radius"]
            obstacle = plt.Circle(
                (obstacle_x, obstacle_y),
                obstacle_radius,
                color="#E31A1C")
            ax.add_patch(obstacle)
            plot_objects["obstacle"] = obstacle

        # Plot Goal
        if show_goal:
            goal_pose = self.goal_point(theta)
            goal_xy = goal_pose.at[:2].get()
            goal = plt.Circle(
                (goal_xy.at[0].get(), goal_xy.at[1].get()),
                self.goal_radius,
                color="#009392",
            )
            ax.add_patch(goal)
            plot_objects["goal"] = goal

        # Plot Slider
        s_th_in_degrees = jnp.degrees(s_th)
        alpha0 = jnp.arctan(s_length / s_width)  # Angle made by diagonal line going from rect lower left to upper right
        half_diagonal_length = jnp.sqrt((s_length / 2) ** 2 + (s_width / 2) ** 2)

        slider = plt.Rectangle(
            (s_x - half_diagonal_length * jnp.cos(alpha0 + s_th), s_y - half_diagonal_length * jnp.sin(alpha0 + s_th)),
            s_width, s_length,
            angle=s_th_in_degrees,
            color='cyan')
        plot_objects["slider"] = slider  # Save slider for later updates in animation.

        ax.add_patch(slider)

        cp = self.contact_point(x)

        # Plot Slider's Geometric Center
        if show_geometric_center:
            geom_center = plt.scatter(s_x, s_y, color='blue', s=10)
            plot_objects["geom_center"] = geom_center

        # Plot Slider's Center of Mass
        if show_CoM:
            th_in_contact_point_frame = s_th - jnp.pi / 2
            rotation_matrix = jnp.array([
                [jnp.cos(th_in_contact_point_frame), -jnp.sin(th_in_contact_point_frame)],
                [jnp.sin(th_in_contact_point_frame), jnp.cos(th_in_contact_point_frame)]
            ])
            CoM = cp + rotation_matrix @ jnp.array([CoM_x, CoM_y])
            CoM.reshape((2, 1))

            CoM_plot = plt.scatter(CoM.at[0].get(), CoM.at[1].get(), color='red', s=10)
            plot_objects["CoM"] = CoM_plot

        # Plot Pusher
        pusher = plt.Circle(
            (cp.at[0].get() - p_radius * jnp.cos(s_th), cp.at[1].get() - p_radius * jnp.sin(s_th)),
            p_radius,
            color="#03DAC6")
        ax.add_patch(pusher)
        plot_objects["pusher"] = pusher

        # Plot Friction Cone Vectors
        if show_friction_cone_vectors:
            # Get Friction Cone Vectors
            friction_cone_vectors = self.friction_cone_extremes()

            # Plot Friction Cone Vectors
            plot_objects["friction_cone_vectors"] = []
            for i, vec in enumerate(friction_cone_vectors):
                vec_clone = vec.copy()
                norm_vec = vec_clone / jnp.linalg.norm(vec_clone)
                scaled_vec = (s_length / 2.0) * norm_vec

                th_in_contact_point_frame = s_th - jnp.pi / 2
                rotation_matrix = jnp.array([
                    [jnp.cos(th_in_contact_point_frame), -jnp.sin(th_in_contact_point_frame)],
                    [jnp.sin(th_in_contact_point_frame), jnp.cos(th_in_contact_point_frame)]
                ])
                rotated_vec = rotation_matrix @ scaled_vec.T

                fcv_plot_i = plt.arrow(cp.at[0].get(), cp.at[1].get(),
                                       rotated_vec.at[0].get(), rotated_vec.at[1].get(),
                                       color="orange", width=0.001)
                plot_objects["friction_cone_vectors"].append(fcv_plot_i)

        # Plot Current Force
        if current_force is not None:
            # Normalize and plot vector of force
            current_force_clone = current_force.copy()
            norm_vec = current_force_clone / jnp.linalg.norm(current_force_clone)
            scaled_vec = (
                jnp.linalg.norm(current_force_clone) / jnp.sqrt(self.max_force**2 + (self.ps_cof * self.max_force)**2)
            ) * (s_length) *  norm_vec

            th_in_contact_point_frame = s_th
            rotation_matrix = jnp.array([
                [jnp.cos(th_in_contact_point_frame), -jnp.cos(th_in_contact_point_frame)],
                [jnp.cos(th_in_contact_point_frame), jnp.cos(th_in_contact_point_frame)]
            ])
            rotated_vec = rotation_matrix @ scaled_vec.T

            cf_plot = plt.arrow(cp.at[0].get(), cp.at[1].get(),
                                rotated_vec.at[0].get(), rotated_vec.at[1].get(),
                                color="green", width=0.001)
            plot_objects["current_force"] = cf_plot

        return plot_objects

    def update_plot_objects(
            self,
            plot_objects: dict,
            x: jnp.array, theta: jnp.array,
            show_geometric_center: bool = False,
            show_CoM: bool = True,
            show_friction_cone_vectors: bool = True,
            current_force: jnp.array = None
    ) -> None:
        # Input Processing
        assert x.shape == (3,), "x must be a tensor of shape (3,); got {}".format(x.shape)
        assert theta.shape == (2,), "theta must be a tensor of shape (2,); got {}".format(theta.shape)

        # Constants
        s_length = self.s_length
        s_width = self.s_width
        p_radius = self.p_radius

        # State
        s_x = x.at[0].get()
        s_y = x.at[1].get()
        s_th = x.at[2].get()

        # Unknown Parameters
        CoM_x = theta.at[0].get()
        CoM_y = theta.at[1].get()

        # Update Slider
        alpha0 = jnp.arctan(s_length / s_width)  # Angle made by diagonal line going from rect lower left to upper right
        half_diagonal_length = jnp.sqrt((s_length / 2) ** 2 + (s_width / 2) ** 2)

        plot_objects["slider"].set_x(s_x - half_diagonal_length * jnp.cos(alpha0 + s_th))
        plot_objects["slider"].set_y(s_y - half_diagonal_length * jnp.sin(alpha0 + s_th))
        plot_objects["slider"].set_angle(jnp.degrees(s_th))

        # Update Geometric Center
        if show_geometric_center:
            # print("showing geometric center = ", show_geometric_center)
            plot_objects["geom_center"].set_offsets((s_x, s_y))

        # Update Center of Mass
        cp = self.contact_point(x)
        if show_CoM:
            th_in_contact_point_frame = s_th - jnp.pi / 2
            rotation_matrix = jnp.array([
                [jnp.cos(th_in_contact_point_frame), -jnp.sin(th_in_contact_point_frame)],
                [jnp.sin(th_in_contact_point_frame), jnp.cos(th_in_contact_point_frame)]
            ])
            CoM = cp + rotation_matrix @ jnp.array([CoM_x, CoM_y])
            plot_objects["CoM"].set_offsets((CoM.at[0].get(), CoM.at[1].get()))

        # Update Pusher
        plot_objects["pusher"].center = (cp.at[0].get() - p_radius * jnp.cos(s_th), cp.at[1].get() - p_radius * jnp.sin(s_th))

        # Update Friction Cone Vectors
        if show_friction_cone_vectors:
            # Get Friction Cone Vectors
            friction_cone_vectors = self.friction_cone_extremes()

            # Plot Friction Cone Vectors
            for i, vec in enumerate(friction_cone_vectors):
                vec_clone = vec
                norm_vec = vec_clone / jnp.linalg.norm(vec_clone)
                scaled_vec = (s_length / 2.0) * norm_vec

                th_in_contact_point_frame = s_th - jnp.pi / 2
                rotation_matrix = jnp.array([
                    [jnp.cos(th_in_contact_point_frame), -jnp.sin(th_in_contact_point_frame)],
                    [jnp.sin(th_in_contact_point_frame), jnp.cos(th_in_contact_point_frame)]
                ])
                rotated_vec = rotation_matrix @ scaled_vec.T

                plot_objects["friction_cone_vectors"][i].set_data(
                    x=cp.at[0].get(), y=cp.at[1].get(),
                    dx=rotated_vec.at[0].get(),
                    dy=rotated_vec.at[1].get())

        # Update Current Force
        if current_force is not None:
            # Normalize and plot vector of force
            current_force_clone = current_force
            norm_vec = current_force_clone / jnp.linalg.norm(current_force_clone)
            scaled_vec = (s_length / 2.0) * norm_vec

            th_in_contact_point_frame = s_th #- jnp.pi / 2
            rotation_matrix = jnp.array([
                [jnp.cos(th_in_contact_point_frame), -jnp.sin(th_in_contact_point_frame)],
                [jnp.cos(th_in_contact_point_frame), jnp.cos(th_in_contact_point_frame)]
            ])
            rotated_vec = rotation_matrix @ scaled_vec.T

            plot_objects["current_force"].set_data(
                x=cp.at[0].get(), y=cp.at[1].get(),
                dx=rotated_vec[0],
                dy=rotated_vec[1])

    def save_animated_trajectory(
            self,
            x_trajectory: jnp.array,
            th: jnp.array,
            f_trajectory: jnp.array,
            limits: Optional[List[List[float]]] = None,
            filename: str = "pusherslider-animation1.mp4",
            hide_axes: bool = True,
            show_obstacle: bool = True,
            show_goal: bool = True,
    ):
        """
        save_animated_trajectory
        Description:
            Animates a trajectory of the pusher-slider system.
        Inputs:
            x_trajectory: A bs x N_traj x 3 tensor containing the trajectory of the system.
            th: A bs x 2 tensor containing the parameters of the system.
            f_trajectory: A bs x (N_traj-1) x 2 tensor containing the trajectory of the forces applied to the system.
            filename: The name of the file to save the animation to.
        """
        # Input Processing
        assert len(
            x_trajectory.shape) == 3, f"x is of the wrong dimension. Received tensor of {len(x_trajectory.shape)} dimensions; expected 3."
        assert x_trajectory.shape[
                   1] == 3, f"Expected state tensor to have second dimension size 3; received {x_trajectory.shape[1]}."

        assert x_trajectory.shape[0] == th.shape[
            0], f"The batch size of x ({x_trajectory.shape[0]}) is different from batch size of theta ({th.shape[0]})."
        assert len(
            th.shape) == 2, f"theta is of the wrong dimension. Received tensor of {len(th.shape)} dimensions; expected 2."
        assert th.shape[1] == 2, f"Expected parameter tensor to have second dimension of size 2; received {th.shape[1]}"

        # Constants
        batch_size = x_trajectory.shape[0]
        N_traj = x_trajectory.shape[2]
        num_frames = N_traj
        dt = self.dt
        max_t = num_frames * dt
        min_t = 0.0
        s = self.nominal_scenario
        obstacle_x, obstacle_y, obstacle_radius = s["obstacle_center_x"], s["obstacle_center_y"], s["obstacle_radius"]
        goal_pose = self.goal_point(th)
        goal_xy = goal_pose.at[:2].get()

        # Create axis limits
        if limits is None:
            limits = []
            x_buffer = self.s_width
            x_limits = [jnp.min(x_trajectory[:, self.S_X, :]) - x_buffer,
                        jnp.max(x_trajectory[:, self.S_X, :]) + x_buffer]
            limits.append(
                x_limits
            )

            y_buffer = self.s_length
            y_limits = [jnp.min(x_trajectory[:, self.S_Y, :] - y_buffer),
                        jnp.max(x_trajectory[:, self.S_Y, :]) + y_buffer]
            limits.append(
                y_limits
            )

            if show_obstacle:
                # Incorporate obstacle in limits
                limits[0][0] = min(limits[0][0], obstacle_x - 2 * obstacle_radius)
                limits[0][1] = max(limits[0][1], obstacle_x + 2 * obstacle_radius)
                limits[1][0] = min(limits[1][0], obstacle_y - 2 * obstacle_radius)
                limits[1][1] = max(limits[1][1], obstacle_y + 2 * obstacle_radius)

            if show_goal:
                # incorporate goal in limits
                goal_radius = 0.1
                limits[0][0] = min(limits[0][0], goal_xy.at[0].get() - 2 * self.goal_radius)
                limits[0][1] = max(limits[0][1], goal_xy.at[0].get() + 2 * self.goal_radius)
                limits[1][0] = min(limits[1][0], goal_xy.at[1].get() - 2 * self.goal_radius)
                limits[1][1] = max(limits[1][1], goal_xy.at[1].get() + 2 * self.goal_radius)

            # print(limits)

            # End result should be a list of lists (e.g., [[0.0, 0.3], [0.0, 0.3]])

        # Create a figure and an axis.
        fig = plt.figure()
        # ax = fig.add_subplot(111)

        # Plot the initial state.
        x0 = x_trajectory[:, :, 0]
        f0 = f_trajectory[:, :, 0]
        plot_collection = self.plot(
            x0, th,
            limits=limits, hide_axes=hide_axes, current_force=f0,
            show_obstacle=show_obstacle, show_goal=show_goal,
        )

        # This function will modify each of the values of the functions above.
        def update(frame_index):

            for batch_index in range(batch_size):
                x_t_bi = x_trajectory[batch_index, :, frame_index]
                f_t_bi = f_trajectory[batch_index, :, frame_index]

                self.update_plot_objects(
                    plot_collection[batch_index],
                    x_t_bi.flatten(), th[batch_index, :].flatten(),
                    current_force=f_t_bi.flatten())

        # Construct the animation, using the update function as the animation
        # director.
        animation = manimation.FuncAnimation(
            fig, update,
            jnp.arange(0, num_frames - 1), interval=5)

        animation.save(filename=filename, fps=15)

    def plot(self, x: jnp.array, theta: jnp.array,
             limits: Optional[List[List[float]]] = None,
             hide_axes: bool = True,
             ax: plt.Axes = None,
             equal_aspect: bool = True,
             show_geometric_center: bool = False,
             show_CoM: bool = True,
             show_friction_cone_vectors: bool = True,
             show_obstacle: bool = True,
             show_goal: bool = True,
             current_force: jnp.array = None) -> List[plt.Figure]:
        """
        plot
        Description
            Plots all pusher sliders in the batch on screen (be careful with this!)
            x:              A bs x 3 tensor containing the states of bs systems.
            theta:          A bs x 2 tensor containing the parameters of bs systems.
            current_force:  A bs x ps.num_controls tensor containing the inputs of bs systems.
        """

        # Constants
        batch_size = x.shape[0]
        s = self.nominal_scenario
        obstacle_x, obstacle_y, obstacle_radius = s["obstacle_center_x"], s["obstacle_center_y"], s["obstacle_radius"]
        goal_pose = self.goal_point(theta)
        goal_xy = goal_pose.at[:2].get()

        # Input Processing
        assert len(
            x.shape) == 2, f"x is of the wrong dimension. Received tensor of {len(x.shape)} dimensions; expected 2."
        assert x.shape[1] == 3, f"Expected state tensor to have second dimension size 3; received {x.shape[1]}."

        assert x.shape[0] == theta.shape[
            0], f"The batch size of x ({x.shape[0]}) is different from batch size of theta ({theta.shape[0]})."

        assert len(
            theta.shape) == 2, f"theta is of the wrong dimension. Received tensor of {len(theta.shape)} dimensions; expected 2."
        assert theta.shape[
                   1] == 2, f"Expected parameter tensor to have second dimension of size 2; received {theta.shape[1]}"

        # Compute Axis Limits
        # Compute Limits
        if limits is None:
            limits = []
            x_buffer = self.s_width
            x_limits = [jnp.min(x[:, self.S_X]) - x_buffer,
                        jnp.max(x[:, self.S_X]) + x_buffer]
            limits.append(
                x_limits
            )

            y_buffer = self.s_length
            y_limits = [jnp.min(x[:, self.S_Y] - y_buffer),
                        jnp.max(x[:, self.S_Y]) + y_buffer]
            limits.append(
                y_limits
            )

            if show_obstacle:
                # Incorporate obstacle in limits
                limits[0][0] = min(limits[0][0], obstacle_x - 2 * obstacle_radius)
                limits[0][1] = max(limits[0][1], obstacle_x + 2 * obstacle_radius)
                limits[1][0] = min(limits[1][0], obstacle_y - 2 * obstacle_radius)
                limits[1][1] = max(limits[1][1], obstacle_y + 2 * obstacle_radius)

            if show_goal:
                # incorporate goal in limits
                goal_radius = 0.1
                limits[0][0] = min(limits[0][0], goal_xy.at[0].get() - 2 * self.goal_radius)
                limits[0][1] = max(limits[0][1], goal_xy.at[0].get() + 2 * self.goal_radius)
                limits[1][0] = min(limits[1][0], goal_xy.at[1].get() - 2 * self.goal_radius)
                limits[1][1] = max(limits[1][1], goal_xy.at[1].get() + 2 * self.goal_radius)

            # End result should be a list of lists (e.g., [[0.0, 0.3], [0.0, 0.3]])

        # Algorithm
        fig = plt.gcf()
        if ax is None:
            ax = fig.add_subplot(111)

        plot_objects_collection = []
        for batch_index in range(batch_size):
            x_bi = x[batch_index, :].flatten()
            theta_bi = theta[batch_index, :].flatten()
            if current_force is not None:
                f_bi = current_force[batch_index, :].flatten()

                plot_objects_bi = self.plot_single(x_bi, theta_bi,
                                                   ax,
                                                   limits=limits, equal_aspect=equal_aspect,
                                                   hide_axes=hide_axes,
                                                   show_geometric_center=show_geometric_center,
                                                   show_CoM=show_CoM,
                                                   show_friction_cone_vectors=show_friction_cone_vectors,
                                                   show_obstacle=show_obstacle and (batch_index == 0),
                                                   show_goal=show_goal and (batch_index == 0),
                                                   current_force=f_bi,
                                                   )
            else:
                plot_objects_bi = self.plot_single(x_bi, theta_bi,
                                                   ax,
                                                   limits=limits, equal_aspect=equal_aspect,
                                                   hide_axes=hide_axes,
                                                   show_geometric_center=show_geometric_center,
                                                   show_CoM=show_CoM,
                                                   show_friction_cone_vectors=show_friction_cone_vectors,
                                                   show_obstacle=show_obstacle and (batch_index == 0),
                                                   show_goal=show_goal and (batch_index == 0),
                                                   )

            plot_objects_collection.append(plot_objects_bi)

        return plot_objects_collection

    def ic_traj_opt(
            self,
            x0: jnp.array,
            horizon: int = 100, num_traj_opt_iters: int = 10000,
            u_step_size=0.01,
            u_clipping=False,
    ) -> tuple[jnp.array, float, float]:
        """
        u_k, opt_end_time - opt_start_time, final_loss = ps.ic_traj_opt(x0, x_star, N=100, N_traj_opt=10000, dt=0.1, u_step_size=0.01)

        Description:
            Computes a trajectory that minimizes the distance of the final point to a point x_star.

        """
        # Constants
        theta = self.theta
        goal = self.goal_point(theta)

        # Define Loss and N-Step Composition Functions
        def NStepCompositionFunction(u: jnp.array):
            """
            NStepCompositionFunction
            Description:
                Computes the N-step composition of the closed loop dynamics.

            """
            # Reshape According to input dimension of ps

            # Compute Composition
            x_t = x0
            for k in range(horizon):
                u_t = u[k, :].reshape((self.n_controls,))
                x_tp1 = x_t + self.closed_loop_dynamics(x_t, u_t) * self.dt

                # Set new variable values for next loop
                x_t = x_tp1

            return x_t

        # Create loss
        def loss(u: jnp.array):
            return jnp.linalg.norm(NStepCompositionFunction(u).at[:2].get() - goal.at[:2].get()) #+ jnp.linalg.norm(u.at[:, 1].get())

        # Define Hill Climbing Procedure
        u_init = jnp.zeros((horizon, self.n_controls))

        grad_L = jax.grad(loss)
        u_k = u_init
        opt_start_time = time.time()
        for k in range(num_traj_opt_iters):
            # At each step measure the loss
            print("Loss at", k, "=", loss(u_k))

            # Update input
            u_kp1 = u_k - u_step_size * grad_L(u_k)

            if u_clipping:
                for i in range(horizon):
                    if u_kp1[i, 0] < -self.ps_cof * u_kp1.at[i, 1].get():
                        u_kp1 = u_kp1.at[i, 0].set(-self.ps_cof * u_kp1.at[i, 1].get())
                    if u_kp1[i, 0] > self.ps_cof * u_kp1.at[i, 1].get():
                        u_kp1 = u_kp1.at[i, 0].set(self.ps_cof * u_kp1.at[i, 1].get())

            # Set variables for next loop iteration
            u_k = u_kp1

        # Finished with optimization
        opt_end_time = time.time()
        final_loss = loss(u_k)

        return u_k, opt_end_time - opt_start_time, final_loss

