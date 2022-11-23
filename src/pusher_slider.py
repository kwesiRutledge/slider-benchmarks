"""
pusher_slider.py
Description:
    Creates a class that represents a Pusher-Slider System.
"""

from jax import jit
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial


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

        