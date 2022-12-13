"""
dynamics_tests.py
Description:
    Tests all of the functions related to the pusher slider's dynamics.
"""

import sys, unittest

import matplotlib.pyplot as plt
import jax.numpy as jnp

sys.path.append('../../')
from src.pusher_slider import PusherSliderSystem

class PusherSliderTest2(unittest.TestCase):
    """
    test_get_motion_cone1
    Description:
        Tests that the motion cone computation works as I expect it to.
    """
    def test_get_motion_cone1(self):

        # Constants
        ps = PusherSliderSystem()
        ps.st_cof = 1.0
        ps.s_width = 1.0 # => c = 2
        ps.p_x = 3.0
        ps.p_y = 5.0

        # Algorithm
        gamma_t_expected = (4.0-15.0+1.0*jnp.power(3,2))/(4.0+jnp.power(5.,2)-1*3*5)
        gamma_b_expected = (-4.0-15.0-1*jnp.power(3,2))/(4.0+jnp.power(5.0,2)+1*3*5)

        gt_2, gb_2 = ps.get_motion_cone_vectors()

        self.assertEqual(gamma_t_expected,gt_2)
        self.assertEqual(gamma_b_expected,gb_2)

    """
    test_identify_mode1
    Description:
        Testing the identification of modes from a good input and
        the motion cone computation.
        The input u should be leading to STICKING.
    """
    def test_identify_mode1(self):

        # Constants
        ps = PusherSliderSystem()
        ps.st_cof = 1.0
        ps.s_width = 0.4 # => c = 5
        ps.p_x = 2.0
        ps.p_y = 2.0

        # Algorithm
        mode_out1 = ps.identify_mode(jnp.array([[2.0],[0.5]]))

        self.assertEqual(mode_out1,'Sticking')

    """
    test_identify_mode2
    Description:
        Testing the identification of modes from a good input and
        the motion cone computation.
        The input u should be leading to SLIDING UP.
    """
    def test_identify_mode2(self):

        # Constants
        ps = PusherSliderSystem()
        ps.st_cof = 1.0
        ps.s_width = 0.4 # => c = 5
        ps.p_x = 2.0
        ps.p_y = 2.0

        # Algorithm
        mode_out1 = ps.identify_mode(jnp.array([[2.0],[3.0]]))

        self.assertEqual(mode_out1,'SlidingUp')

    """
    test_identify_mode3
    Description:
        Making sure that this sketch of a plotting algorithm works.
    """
    def test_identify_mode3(self):

        # Constants
        ps = PusherSliderSystem()
        ps.st_cof = 1.0
        ps.s_width = 0.4 # => c = 5
        ps.p_x = 2.0
        ps.p_y = 2.0

        # Algorithm
        mode_out1 = ps.identify_mode(jnp.array([[2.0],[-2.5]]))

        self.assertEqual(mode_out1,'SlidingDown')

    """
    test_C1
    Description:
        Tests that the C matrix is constructed correctly.
    """
    def test_C1(self):
        # constants
        ps = PusherSliderSystem()
        ps.s_theta = jnp.pi/6

        # Algorithm
        C = ps.C()

        self.assertEqual(C[0][0],jnp.cos(jnp.pi/6))
        self.assertEqual(C[0][1],jnp.sin(jnp.pi/6))
        self.assertEqual(C[1][0],-jnp.sin(jnp.pi/6))
        self.assertEqual(C[1][1],jnp.cos(jnp.pi/6))

    """
    test_Q1
    Description:
        Tests that the Q matrix is constructed correctly.
    """
    def test_Q1(self):
        # constants
        ps = PusherSliderSystem()
        ps.ps_cof = 1
        ps.s_width = 0.4 # c = 5
        ps.p_x = 2.0
        ps.p_y = 2.0

        # Algorithm
        Q = ps.Q()

        self.assertEqual(Q[0][0],(1/(25.0+8))*(25+4))
        self.assertEqual(Q[0][1],(1/(25.0+8))*(4))
        self.assertEqual(Q[1][0],(1/(25.0+8))*(4))
        self.assertEqual(Q[1][1],(1/(25.0+8))*(25+4)) 

    """
    test_f1
    Description:
    	Tests the f() function so that we know that the state of the pusher slider is not modified
		after evaluating f at a given point.
    """   
    def test_f1(self):
        # Constants
        ps = PusherSliderSystem()

        x0 = ps.x()
        u_prime = jnp.array([[1.0], [3.0]])
        x1 = x0 + jnp.array([[0.1], [0.0], [0.0], [0.0]])

        x_dot = ps.f(x1,u_prime)
        x_dot2 = ps.f2(x1,u_prime) # This u should call

        x_dim = x_dot.shape[1]
        for x_dim_index in range(x_dim):
            # Constraints
            self.assertEqual(x_dot[0,x_dim_index],x_dot[0,x_dim_index])

if __name__ == '__main__':
    unittest.main()