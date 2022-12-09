"""
pusher_slider.py
Description:
    Creates a class that represents a Pusher-Slider System.
"""

from jax import jit
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
import matplotlib.animation as manimation



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
        mu = self.ps_cof # TODO: Which coefficient of friction is this supposed to be?

        gamma_t = ( mu*jnp.power(c,2) - self.p_x * self.p_y + mu * jnp.power(self.p_x,2) ) / \
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

    
    """
    C
    Description:
        Creates the rotation matrix used in the pusher slider system's dynamics.
        Note: This is NOT the C matrix from the common linear system's y = Cx + v.
    """
    def C(self)->jnp.array:
        # Constants
        theta = self.s_theta

        # Algorithm
        return jnp.array([
            [jnp.cos(theta),jnp.sin(theta)],
            [-jnp.sin(theta),jnp.cos(theta)]
        ])

    """
    Q
    Description:
        Creates the Q matrix defined in the equations of motion for the pusher slider.
    """
    def Q(self)->jnp.array:
        # Constants
        g = 10.0
        f_max = self.st_cof * self.s_mass * g
        m_max = self.st_cof * self.s_mass * g * (self.s_width/2.0)  # The last term is meant to come from
                                                                    # a sort of mass distribution/moment calculation.
        c = f_max / m_max
        p_x = self.p_x
        p_y = self.p_y

        # Algorithm
        return (1.0/(jnp.power(c,2) + jnp.power(p_x,2) + jnp.power(p_y,2))) * \
            jnp.array([
                [jnp.power(c,2)+jnp.power(p_x,2), p_x*p_y],
                [p_x*p_y, jnp.power(c,2)+jnp.power(p_y,2)]
            ])

    """
    set_state
    Description:
        Sets the state of the pusher slider according to the state x
        where
                [   s_x   ]
            x = [   s_y   ]
                [ s_theta ]
                [   p_y   ]

    """
    def set_state(self,x):
        # Constants

        # Algorithm
        self.s_x = x[0,0]
        self.s_y = x[1,0]
        self.s_theta = x[2,0]
        self.p_y = x[3,0]

    """
    set_input
    Description:
        Gets the current input, where the input of the pusher slider x is
    
           u = [ v_n ]
                [ v_t ]
    Usage:
        ps.set_input(u)
    """
    def set_input(self,u):
        self.v_n = u[0][0]
        self.v_t = u[1][0]

    """
    get_state
    Description:
        Gets the state of the pusher slider according to the state x
        where
                [   s_x   ]
            x = [   s_y   ]
                [ s_theta ]
                [   p_y   ]

    Usage:
        x = ps.get_state()
    """
    def get_state(self):
        return jnp.array([[self.s_x],[self.s_y],[self.s_theta],[self.p_y]])

    """
    get_input
    Description:
        Gets the current input, where the input of the pusher slider x is
            u = [ v_n ]
                [ v_t ]
    Usage:
        u = ps.get_input()
    """
    def get_input(self):
        return jnp.array([[self.v_n],[self.v_t]])

    """
    x
    Description:
        Alias for the get_state function.
    Usage:
        x = ps.x()
    """
    def x(self):
        return self.get_state()

    """
    u
    Description:
        Alias for the get_input function.
    Usage:
        u = ps.u()
    """
    def u(self):
        return self.get_input()

    """
    f1
    Description:
        Continuous dynamics of the sticking mode of contact between pusher and slider.
    """
    # @partial(jit, static_argnums=(0,))
    def f1(self,x,u):
        # Constants
        self.set_state(x)
        self.set_input(u)
        C0 = self.C()
        Q0 = self.Q()

        g = 10.0
        f_max = self.st_cof * self.s_mass * g
        m_max = self.st_cof * self.s_mass * g * (self.s_width/2.0)  # The last term is meant to come from
                                                                    # a sort of mass distribution/moment calculation.
        c = f_max / m_max
        p_x = self.p_x
        p_y = self.p_y

        # Algorithm
        b1 = jnp.array([
            [ - p_y / (jnp.power(c,2)+jnp.power(p_x,2)+jnp.power(p_y,2)) , p_x ]
        ])

        c1 = jnp.array([[0.0,0.0]])

        P1 = jnp.eye(2)

        #       = [ C0 * Q0 * P1 ]
        # dxdt  = [      b1      ] * u
        #       = [      c1      ]

        return jnp.vstack(
            (C0.T.dot( Q0.dot(P1) ),b1,c1)
        ).dot(u)

    """
    f2
    Description:
        Continuous dynamics of the SlidingUp mode of contact between pusher and slider.
    """
    # @partial(jit, static_argnums=(0,))
    def f2(self,x,u):
        #Constants
        self.set_state(x)
        self.set_input(u)
        C0 = self.C()
        Q0 = self.Q()

        g = 10.0
        f_max = self.st_cof * self.s_mass * g
        m_max = self.st_cof * self.s_mass * g * (self.s_width/2.0)  # The last term is meant to come from
                                                                    # a sort of mass distribution/moment calculation.
        c = f_max / m_max
        p_x = self.p_x
        p_y = self.p_y

        gamma_t, gamma_b = self.get_motion_cone_vectors()

        # Algorithm
        b2 = jnp.array([
            [(-p_y+gamma_t*p_x)/(jnp.power(c,2)+jnp.power(p_x,2)+jnp.power(p_y,2)),0.0]
        ])

        c2 = jnp.array([[-gamma_t,0.0]])

        P2 = jnp.array([
            [1.0,0.0],
            [gamma_t,0]
        ])

        #       = [ C0 * Q0 * P2 ]
        # dxdt  = [      b2      ] * u
        #       = [      c2      ]

        return jnp.vstack(
            (C0.dot( Q0.dot(P2) ),b2,c2)
        ).dot(u)

    """
    f3
    Description:
        Continuous dynamics of the SlidingDown mode of contact between pusher and slider.
    """
    # @partial(jit, static_argnums=(0,))
    def f3(self,x,u):
        #Constants
        self.set_state(x)
        self.set_input(u)
        C0 = self.C()
        Q0 = self.Q()

        g = 10
        f_max = self.st_cof * self.s_mass * g
        m_max = self.st_cof * self.s_mass * g * (self.s_width/2.0)  # The last term is meant to come from
                                                                    # a sort of mass distribution/moment calculation.
        c = f_max / m_max
        p_x = self.p_x
        p_y = self.p_y

        gamma_t, gamma_b = self.get_motion_cone_vectors()

        # Algorithm
        b3 = jnp.array([
            [(-p_y+gamma_b*p_x)/(jnp.power(c,2)+jnp.power(p_x,2)+jnp.power(p_y,2)),0]
        ])

        c3 = jnp.array([[-gamma_b,0.0]])

        P3 = jnp.array([
            [1.0,0.0],
            [gamma_b,0.0]
        ])

        #       = [ C0 * Q0 * P3 ]
        # dxdt  = [      b3      ] * u
        #       = [      c3      ]

        return jnp.vstack(
            (C0.dot( Q0.dot(P3) ),b3,c3)
        ).dot(u)

    """
    f
    Description:
        Defines dynamics of the hybrid pusher-slider system.
    Usage:
        dxdt = ps.f(x,u)
    """
    def f(self,x,u):
        # Constants

        # Algorithm
        curr_mode = self.identify_mode(u)

        if curr_mode == 'Sticking':
            return self.f1(x,u)
        elif curr_mode == 'SlidingUp':
            return self.f2(x,u)
        elif curr_mode == 'SlidingDown':
            return self.f3(x,u)
        else:
            raise(Exception("There was a problem with identifying the current mode! Unexpected mode = " + curr_mode))


    """
    convert_plan_to_video
    Description:
        Takes a plan (in the form of trajectories of a Pusher Slider System) and converts it into a video.    
    """
    def convert_plan_to_video(self,x_trajectory,t:jnp.array,hide_axes=False):
        # Constants
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(
            title=movie_title,
            artist='Matplotlib',
            comment='A Pusher-slider trajectory of length ' + x_trajectory.shape[1] + '!'
        )
        writer = FFMpegWriter(fps=15, metadata=metadata)

        num_frames = 100 # What is this for??

        # Define bounds for the window using  pusher-slider trajectory
        x_min = jnp.min(
            jnp.hstack( x_trajectory[0,:] , x_trajectory[3,:] )
        )

        # Algorithm
        # =========
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if hide_axes:
            ax.axis('off')

        if limits is not None:
            plt.xlim(limits[0])
            plt.ylim(limits[1])
        else:
            vertices = np.concatenate(vertices, axis=0)
            xmin, ymin = vertices.min(axis=0)
            xmax, ymax = vertices.max(axis=0)
            plt.xlim([xmin - 0.1, xmax + 0.1])
            plt.ylim([ymin - 0.1, ymax + 0.1])

        if equal_aspect:
            plt.gca().set_aspect('equal', adjustable='box')

        if pwl_plans is None or pwl_plans[0] is None:
            plt.show()
            return

        if len(pwl_plans) <= 4:
            colors = ['k', np.array([153, 0, 71]) / 255, np.array([6, 0, 153]) / 255, np.array([0, 150, 0]) / 255]
        else:
            cmap = plt.get_cmap('tab10')
            colors = [cmap(i) for i in np.linspace(0, 0.85, len(pwl_plans))]

        # num_agents
        team_positions_at_t = np.zeros((2, num_agents))
        agent_circles = []
        for i in range(num_agents):
            PWL = pwl_plans[i]
            x_t = get_state_at_t(0.0, PWL)
            team_positions_at_t[:, i] = x_t
            # print(team_positions_at_t)
            agent_circles.append(
                plt.Circle((team_positions_at_t[0, i], team_positions_at_t[1, i]), size_list[i], color=colors[i])
            )
            ax.add_patch(agent_circles[-1])

        for i in range(len(pwl_plans)):
            PWL = pwl_plans[i]
            ax.plot([P[0][0] for P in PWL], [P[0][1] for P in PWL], '-', color=colors[i])

            ax.plot(PWL[-1][0][0], PWL[-1][0][1], '*', color=colors[i])
            # print(PWL[0][0][0])
            # print(size_list[i])
            # ax.plot(PWL[0][0][0], PWL[0][0][1], 'o', color = colors[i])

        # Plot the Team Plan
        if show_team_plan:
            for P in team_plan:
                ax.add_patch(
                    plt.Circle((P[0][0], P[0][1]), team_radius, color='m', alpha=0.2)
                )

        # Plot the team plan over time
        P0 = team_plan[0]
        team_circle = plt.Circle((P0[0][0], P0[0][1]), team_radius, color='m', alpha=0.2)
        if show_moving_team_radius:
            ax.add_patch(
                team_circle
            )

        # This function will modify each of the values of the functions above.
        def update(frame_number):
            t = (frame_number / num_frames) * (max_t - min_t) + min_t
            # print(t)
            for i in range(num_agents):
                plan_i = pwl_plans[i]
                # print(plan_i)
                x_t = get_state_at_t(t, plan_i)
                team_positions_at_t[:, i] = x_t
                agent_circles[i].set(
                    center=x_t,
                )

            # If we want to show the team circle moving, then update it here
            if show_moving_team_radius:
                team_center_t = get_state_at_t(t, team_plan)
                team_circle.set(
                    center=team_center_t
                )

        # Construct the animation, using the update function as the animation
        # director.
        animation = manimation.FuncAnimation(fig, update, np.arange(1, num_frames), interval=25)
        animation.save(filename=filename, fps=15)

        # # Algorithm
        # for t in np.linspace(min_t,max_t,num_frames):
        #     fig_t = plot_single_frame_of_team_plan(
        #         t, pwl_plans=pwl_plans, team_plan=team_plan, plot_tuples=plot_tuples, team_radius=team_radius, size_list=size_list, equal_aspect=equal_aspect, limits=limits, show_team_plan=True        )

        #     writer.saving(fig_t,filename,20)
        #     writer.grab_frame()
