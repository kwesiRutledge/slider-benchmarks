"""
pusher_slider_sticking_velocity_input.py
Description:
    Creates a class that represents a Pusher-Slider System
    where we assume that the velocity of the pusher is the input.
"""

from jax import jit
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
import matplotlib.animation as manimation

class PusherSliderStickingVelocityInputSystem(object):
    """
    ps = PusherSliderStickingVelocityInputSystem()
    Description:
        Creates a class that represents a Pusher-Slider System
        where we assume that the velocity of the pusher is the input
        and the slider is sticking to the pusher.
    """

    # Constants
    S_X = 0
    S_Y = 1
    S_THETA = 2

    # Input Names
    V_N = 0
    V_T = 1

    N_DIMS = 3
    N_CONTROLS = 2

    def __init__(self):
        """
        __init__
        Description:
            For the sake of cleanliness, I will not include the option to
            initialize parameters with input values. For now, do that outside
            of this initializer.
        """
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
        self.p_y = 0.0

        # Define Initial Input
        self.v_n = 0.01 # Velocity normal to the slider
        self.v_t = 0.03 # Velocity tangential to the slider

    @property
    def n_dims(self):
        return PusherSliderStickingVelocityInputSystem.N_DIMS

    @property
    def n_controls(self):
        return PusherSliderStickingVelocityInputSystem.N_CONTROLS

    def plot(self, fig_in: plt.figure)->None:
        """
        plot
        Description:
            Plots the pusher-slider system onto a new figure.
        """

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
            plt.Circle(
                (circle_center[0], circle_center[1]),
                self.p_radius, color=pusher_color, alpha=0.2,
            )
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
        mu = self.ps_cof # TODO: Which coefficient of friction is this supposed to be?

        gamma_t = ( mu*jnp.power(c,2) - self.p_x * self.p_y + mu * jnp.power(self.p_x,2) ) / \
            ( jnp.power(c,2) + jnp.power(self.p_y,2) - mu * self.p_x*self.p_y )

        gamma_b = ( - mu*jnp.power(c,2) - self.p_x * self.p_y - mu * jnp.power(self.p_x,2) )/ \
            ( jnp.power(c,2) + jnp.power(self.p_y,2) + mu * self.p_x*self.p_y )

        return gamma_t, gamma_b


    def C(self, x: jnp.array) -> jnp.array:
        """
        rot = ps.C(x)
        Description:
            Creates the rotation matrix used in the pusher slider system's dynamics.
            Note: This is NOT the C matrix from the common linear system's y = Cx + v.
        """
        # Constants
        theta = x.at[2].get()

        # Algorithm
        return jnp.array([
            [jnp.cos(theta), jnp.sin(theta)],
            [-jnp.sin(theta), jnp.cos(theta)]
        ])

    def Q(self, x: jnp.array) -> jnp.array:
        """
        q = ps.Q(xx)
        Description:
            Creates the Q matrix defined in the equations of motion for the pusher slider.
        """

        # Constants
        g = 10.0
        f_max = self.st_cof * self.s_mass * g
        m_max = self.st_cof * self.s_mass * g * (2.0*self.s_width/3.0)  # The last term is meant to come from
                                                                    # a sort of mass distribution/moment calculation.
        c = m_max / f_max

        # State
        p_x = self.p_x
        p_y = self.p_y

        # Algorithm
        return (1.0/(jnp.power(c, 2) + jnp.power(p_x, 2) + jnp.power(p_y,2))) * \
            jnp.array([
                [jnp.power(c, 2)+jnp.power(p_x, 2), p_x*p_y],
                [p_x*p_y, jnp.power(c, 2)+jnp.power(p_y, 2)]
            ])

    # @partial(jit, static_argnums=(0,))
    def f1(self, x: jnp.array, u: jnp.array) -> jnp.array:
        """
        dxdt = ps.f1(x,u)
        Description:
            Continuous dynamics of the sticking mode of contact between pusher and slider.
        """

        # Input Processing
        assert x.shape == (self.n_dims,)
        assert u.shape == (self.n_controls,)

        # Constants
        C0 = self.C(x)
        Q0 = self.Q(x)

        g = 10.0
        f_max = self.st_cof * self.s_mass * g
        m_max = self.st_cof * self.s_mass * g * (self.s_width/2.0)  # The last term is meant to come from
                                                                    # a sort of mass distribution/moment calculation.
        c = m_max / f_max
        p_x = self.p_x
        p_y = self.p_y

        # Algorithm
        b1 = jnp.array([
            [- p_y / (jnp.power(c, 2)+jnp.power(p_x, 2)+jnp.power(p_y, 2)), p_x / (jnp.power(c, 2)+jnp.power(p_x, 2)+jnp.power(p_y, 2))]
        ])

        # c1 = jnp.array([[0.0, 0.0]])

        P1 = jnp.eye(2)

        #       = [ C0 * Q0 * P1 ]
        # dxdt  = [      b1      ] * u

        return jnp.vstack(
            (C0.T.dot(Q0.dot(P1)), b1)
        ).dot(u)

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
            scaled_vec = (s_length / 2.0) * norm_vec

            th_in_contact_point_frame = s_th - jnp.pi / 2
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
        plot_objects["pusher"].center = (
        cp.at[0].get() - p_radius * jnp.cos(s_th), cp.at[1].get() - p_radius * jnp.sin(s_th))

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

            th_in_contact_point_frame = s_th - jnp.pi / 2
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
