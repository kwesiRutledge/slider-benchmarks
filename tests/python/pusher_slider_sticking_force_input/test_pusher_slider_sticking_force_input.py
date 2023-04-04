"""
test_pusher_slider_sticking_force_input.py
Description:

"""

import unittest

import jax.numpy as jnp

import sys
sys.path.append('../../../')
from src.python.pusher_slider_sticking_force_input import PusherSliderStickingForceInputSystem

class Test_PusherSliderStickingForceInput(unittest.TestCase):
    def test_friction_cone_extremes1(self):
        # Constants
        test_cof = 1.3
        scenario = {
            "obstacle_center_x": 0.0,
            "obstacle_center_y": 0.0,
            "obstacle_radius": 0.1,
        }
        ps = PusherSliderStickingForceInputSystem(scenario, ps_cof=test_cof)
        f_minus, f_plus = ps.friction_cone_extremes()

        assert f_plus.shape == (2,)
        assert f_minus.shape == (2,)

        assert f_minus[0] == test_cof, f"Expected f_minus[0] = {test_cof}, Actual: {f_minus[0]}"
        assert f_plus[0] == -test_cof, f"Expected f_plus[0] = {-test_cof}, Actual: {f_plus[0]}"


    def test_goal_state1(self):
        # Constants
        scenario = {
            "obstacle_center_x": 0.0,
            "obstacle_center_y": 0.0,
            "obstacle_radius": 0.1,
        }
        aps = PusherSliderStickingForceInputSystem(scenario)
        theta = jnp.array([[0.0, aps.s_length/2.0]])

        # Create Goal State
        x_goal = aps.goal_state(theta)

        assert x_goal.shape == (1, 3)

        assert jnp.isclose(x_goal.at[0, 2].get(), 0.0), f"Expected x_goal[0, 2] = 0.0, received: {x_goal.at[0, 2]}"

    def test__f1(self):
        # Constants
        scenario = {
            "obstacle_center_x": 0.0,
            "obstacle_center_y": 0.0,
            "obstacle_radius": 0.1,
        }
        aps = PusherSliderStickingForceInputSystem(scenario)
        x = jnp.array([.0, 0.0, 0.0])
        theta = jnp.array([0.0, aps.s_length/2.0])

        # Compute f
        f = aps._f(x)

        assert f.shape == (3,), f"Expected f.shape = (3,), received: {f.shape}"

        assert jnp.all(
            jnp.isclose(f, jnp.zeros((1, 3, 1)))
        ), f"Expected f = 0.0, received: {f.get()}"

    def test__F1(self):
        # Constants
        scenario = {
            "obstacle_center_x": 0.0,
            "obstacle_center_y": 0.0,
            "obstacle_radius": 0.1,
        }
        aps = PusherSliderStickingForceInputSystem(scenario)
        x = jnp.array([0.0, 0.0, 0.0])

        # Compute f
        F = aps._F(x)

        assert F.shape == (3, 2), f"Expected F.shape = (3, 2), received: {F.shape}"

        assert jnp.all(
            jnp.isclose(F, jnp.zeros((3, 2)))
        ), f"Expected F = 0.0, received: {F.get()}"

    def test__g1(self):
        """
        test__g1
        Description:
            Tests the function _g for the Adaptive Pusher Slider.
        """
        # Constants
        scenario = {
            "obstacle_center_x": 0.0,
            "obstacle_center_y": 0.0,
            "obstacle_radius": 0.1,
        }
        aps = PusherSliderStickingForceInputSystem(scenario)
        s_x = 0.0
        s_y = 1.0
        s_theta = jnp.pi/3.0
        x = jnp.array([s_x, s_y, s_theta])

        f_max, tau_max = aps.limit_surface_bounds()
        a = (1 / 10.0) * (1 / (f_max ** 2))
        b = (1 / 10.0) * (1 / (tau_max ** 2))

        # Compute g
        g = aps._g(x)

        assert g.shape == (3, 2), f"Expected F.shape = (3, 2), received: {g.shape}"

        assert not jnp.all(
            jnp.isclose(g, jnp.zeros((3, 2)))
        ), f"Expected g =/= 0.0, received: {g.get()}"

        assert jnp.isclose(
            g.at[PusherSliderStickingForceInputSystem.S_THETA, PusherSliderStickingForceInputSystem.F_X].get(),
            0.0,
        ), f"Expected g[{PusherSliderStickingForceInputSystem.S_THETA}, {PusherSliderStickingForceInputSystem.F_X}] = 0.0, received: {g.at[PusherSliderStickingForceInputSystem.S_THETA, PusherSliderStickingForceInputSystem.F_X].get()}"

        assert jnp.isclose(
            g.at[PusherSliderStickingForceInputSystem.S_THETA, PusherSliderStickingForceInputSystem.F_Y].get(),
            0.0,
        ), f"Expected g[{PusherSliderStickingForceInputSystem.S_THETA}, {PusherSliderStickingForceInputSystem.F_Y}] = 0.0, received: {g.at[PusherSliderStickingForceInputSystem.S_THETA, PusherSliderStickingForceInputSystem.F_Y].get()}"

        # Investigate nonzero elements
        assert jnp.isclose(
            g.at[PusherSliderStickingForceInputSystem.S_X, PusherSliderStickingForceInputSystem.F_X].get(),
            jnp.cos(s_theta) * a,
        ), f"Expected g[{PusherSliderStickingForceInputSystem.S_X}, {PusherSliderStickingForceInputSystem.F_X}] = {jnp.cos(s_theta) * a}, received: {g.at[PusherSliderStickingForceInputSystem.S_X, PusherSliderStickingForceInputSystem.F_X].get()}"
        assert jnp.isclose(
            g.at[PusherSliderStickingForceInputSystem.S_X, PusherSliderStickingForceInputSystem.F_Y].get(),
            -jnp.sin(s_theta) * a,
        ), f"Expected g[{PusherSliderStickingForceInputSystem.S_X}, {PusherSliderStickingForceInputSystem.F_Y}] = {-jnp.sin(s_theta) * a}, received: {g.at[PusherSliderStickingForceInputSystem.S_X, PusherSliderStickingForceInputSystem.F_Y].get()}"

        assert jnp.isclose(
            g.at[PusherSliderStickingForceInputSystem.S_Y, PusherSliderStickingForceInputSystem.F_X].get(),
            jnp.sin(s_theta) * a,
        ), f"Expected g[{PusherSliderStickingForceInputSystem.S_Y}, {PusherSliderStickingForceInputSystem.F_X}] = {jnp.sin(s_theta) * a}, received: {g.at[PusherSliderStickingForceInputSystem.S_Y, PusherSliderStickingForceInputSystem.F_X].get()}"

        assert jnp.isclose(
            g.at[PusherSliderStickingForceInputSystem.S_Y, PusherSliderStickingForceInputSystem.F_Y].get(),
            jnp.cos(s_theta) * a,
        ), f"Expected g[{PusherSliderStickingForceInputSystem.S_Y}, {PusherSliderStickingForceInputSystem.F_Y}] = {jnp.cos(s_theta) * a}, received: {g.at[PusherSliderStickingForceInputSystem.S_Y, PusherSliderStickingForceInputSystem.F_Y].get()}"

    def test__G1(self):
        # Constants
        scenario = {
            "obstacle_center_x": 0.0,
            "obstacle_center_y": 0.0,
            "obstacle_radius": 0.1,
        }
        aps = PusherSliderStickingForceInputSystem(scenario)
        s_x = 0.0
        s_y = 1.0
        s_theta = jnp.pi / 3.0
        x = jnp.array([s_x, s_y, s_theta])

        f_max, tau_max = aps.limit_surface_bounds()
        a = (1 / 10.0) * (1 / (f_max ** 2))
        b = (1 / 10.0) * (1 / (tau_max ** 2))

        # Compute g
        G = aps._G(x)

        batch_size = x.shape[0]

        assert G.shape == (aps.n_dims, aps.n_controls, aps.n_params),\
            f"Expected G.shape = ({batch_size}, {aps.n_dims}, {aps.n_controls}, {aps.n_params}), received: {G.shape}"

        # Check where zeros should be
        assert jnp.isclose(
            G.at[
                PusherSliderStickingForceInputSystem.S_THETA,
                PusherSliderStickingForceInputSystem.F_X,
                PusherSliderStickingForceInputSystem.C_X,
            ].get(),
            0.0,
        ), f"Expected G[{PusherSliderStickingForceInputSystem.S_THETA}, {PusherSliderStickingForceInputSystem.F_X}, {PusherSliderStickingForceInputSystem.C_X}] = 0.0, received: {G.at[PusherSliderStickingForceInputSystem.S_THETA, PusherSliderStickingForceInputSystem.F_X, PusherSliderStickingForceInputSystem.C_X].get()}"

        assert jnp.isclose(
            G.at[
                PusherSliderStickingForceInputSystem.S_THETA,
                PusherSliderStickingForceInputSystem.F_Y,
                PusherSliderStickingForceInputSystem.C_Y,
            ].get(),
            0.0,
        ), f"Expected G[{PusherSliderStickingForceInputSystem.S_THETA}, {PusherSliderStickingForceInputSystem.F_Y}, {PusherSliderStickingForceInputSystem.C_Y}] = 0.0, received: {G.at[PusherSliderStickingForceInputSystem.S_THETA, PusherSliderStickingForceInputSystem.F_Y, PusherSliderStickingForceInputSystem.C_Y].get()}"

        # Check where NON-zeros should be
        assert jnp.isclose(
            G.at[
                PusherSliderStickingForceInputSystem.S_THETA,
                PusherSliderStickingForceInputSystem.F_X,
                PusherSliderStickingForceInputSystem.C_Y,
            ].get(),
            -b,
        ), f"Expected G[{PusherSliderStickingForceInputSystem.S_THETA}, {PusherSliderStickingForceInputSystem.F_X}, {PusherSliderStickingForceInputSystem.C_Y}] = {-b}, received: {G.at[PusherSliderStickingForceInputSystem.S_THETA, PusherSliderStickingForceInputSystem.F_X, PusherSliderStickingForceInputSystem.C_Y].get()}"

        assert jnp.isclose(
            G.at[
                PusherSliderStickingForceInputSystem.S_THETA,
                PusherSliderStickingForceInputSystem.F_Y,
                PusherSliderStickingForceInputSystem.C_X,
            ].get(),
            b,
        ), f"Expected G[{PusherSliderStickingForceInputSystem.S_THETA}, {PusherSliderStickingForceInputSystem.F_Y}, {PusherSliderStickingForceInputSystem.C_X}] = {b}, received: {G.at[PusherSliderStickingForceInputSystem.S_THETA, PusherSliderStickingForceInputSystem.F_Y, PusherSliderStickingForceInputSystem.C_X].get()}"

    def test_input_gain_matrix1(self):
        """
        test_input_gain_matrix1
        Description:
            - Test the input gain matrix for the system
        """
        # Constants
        scenario = {
            "obstacle_center_x": 0.0,
            "obstacle_center_y": 0.0,
            "obstacle_radius": 0.1,
        }
        aps = PusherSliderStickingForceInputSystem(scenario)
        s_x = 0.0
        s_y = 1.0
        s_theta = jnp.pi / 3.0
        x = jnp.array([s_x, s_y, s_theta])

        c_x = 0.0
        c_y = aps.s_width/2.0
        theta = jnp.array([c_x, c_y])

        f_max, tau_max = aps.limit_surface_bounds()
        a = (1 / 10.0) * (1 / (f_max ** 2))
        b = (1 / 10.0) * (1 / (tau_max ** 2))

        # Compute g
        G = aps._G(x)
        g = aps._g(x)

        # Compute input gain matrix
        K = aps.input_gain_matrix(x, theta)

        assert K.shape == (aps.n_dims, aps.n_controls),\
            f"Expected K.shape = ({aps.n_dims}, {aps.n_controls}), received: {K.shape}"

        # Check where some nonzeros should be
        # K[2, 0] = nonzero!
        expected_entry = g.at[
            PusherSliderStickingForceInputSystem.S_THETA,
            PusherSliderStickingForceInputSystem.F_X,
        ].get() + theta.at[0].get() * G.at[
            PusherSliderStickingForceInputSystem.S_THETA,
            PusherSliderStickingForceInputSystem.F_X,
            PusherSliderStickingForceInputSystem.C_X,
        ].get() + theta.at[1].get() * G.at[
            PusherSliderStickingForceInputSystem.S_THETA,
            PusherSliderStickingForceInputSystem.F_X,
            PusherSliderStickingForceInputSystem.C_Y,
        ].get()

        assert jnp.isclose(
            K.at[
                PusherSliderStickingForceInputSystem.S_THETA,
                PusherSliderStickingForceInputSystem.F_X,
            ].get(),
            expected_entry,
        ), "Expected K[{}, {}] = {}, received: {}".format(
            PusherSliderStickingForceInputSystem.S_THETA,
            PusherSliderStickingForceInputSystem.F_X,
            expected_entry,
            K.at[
                PusherSliderStickingForceInputSystem.S_THETA,
                PusherSliderStickingForceInputSystem.F_X
            ].get(),
        )

        # K[0,0] = nonzero!
        expected_entry = g.at[
             PusherSliderStickingForceInputSystem.S_X,
             PusherSliderStickingForceInputSystem.F_X,
         ].get() + theta.at[0].get() * G.at[
             PusherSliderStickingForceInputSystem.S_X,
             PusherSliderStickingForceInputSystem.F_X,
             PusherSliderStickingForceInputSystem.C_X,
         ].get() + theta.at[1].get() * G.at[
             PusherSliderStickingForceInputSystem.S_X,
             PusherSliderStickingForceInputSystem.F_X,
             PusherSliderStickingForceInputSystem.C_Y,
         ].get()

        assert jnp.isclose(
            K.at[
                PusherSliderStickingForceInputSystem.S_X,
                PusherSliderStickingForceInputSystem.F_X,
            ].get(),
            expected_entry,
        ), "Expected K[{}, {}] = {}, received: {}".format(
            PusherSliderStickingForceInputSystem.S_X,
            PusherSliderStickingForceInputSystem.F_X,
            expected_entry,
            K.at[
                PusherSliderStickingForceInputSystem.S_X,
                PusherSliderStickingForceInputSystem.F_X
            ].get(),
        )

    def test_control_affine_dynamics1(self):
        """
        test_control_affine_dynamics1
        Description:
            - Test the control affine dynamics function for the system
        """

        # Constants
        scenario = {
            "obstacle_center_x": 0.0,
            "obstacle_center_y": 0.0,
            "obstacle_radius": 0.1,
        }
        aps = PusherSliderStickingForceInputSystem(scenario)
        s_x = 0.0
        s_y = 1.0
        s_theta = jnp.pi / 3.0
        x = jnp.array([s_x, s_y, s_theta])

        c_x = 0.0
        c_y = aps.s_width / 2.0
        theta = jnp.array([c_x, c_y])

        # scenario = {
        #     "obstacle_center_x": 0.0,
        #     "obstacle_center_y": 0.0,
        #     "obstacle_radius": 0.1,
        # }

        # Test
        f, g = aps.control_affine_dynamics(x)

        assert f.shape == aps._f(x).shape, f"Expected f.shape = {aps._f(x).shape}, received: {f.shape}"
        assert g.shape == aps._g(x).shape, f"Expected g.shape = {aps._g(x).shape}, received: {g.shape}"

        assert jnp.allclose(
            f,
            aps._f(x) + aps._F(x) @ theta
        ), "Expected f = {}, received: {}".format(aps._f(x) + aps._F(x) @ theta, f)
        assert jnp.allclose(
            g,
            aps.input_gain_matrix(x, theta),
        ), "Expected g = {}, received: {}".format(aps.input_gain_matrix(x, theta), g)


if __name__ == '__main__':
    unittest.main()
