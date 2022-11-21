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

    

if __name__ == '__main__':
    unittest.main()