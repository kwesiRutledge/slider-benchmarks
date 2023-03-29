"""
test_pusher_slider_sticking_force_input.py
Description:

"""

import unittest

import jax.numpy as jnp

import sys
sys.path.append('../../')
from src.python.pusher_slider_sticking_force_input import AdaptivePusherSliderStickingForceInputSystem

class Test_PusherSliderStickingForceInput(unittest.TestCase):
    def test_friction_cone_extremes1(self):
        # Constants
        test_cof = 1.3
        ps = AdaptivePusherSliderStickingForceInputSystem(ps_cof=test_cof)
        f_minus, f_plus = ps.friction_cone_extremes()

        assert f_plus.shape == (2,)
        assert f_minus.shape == (2,)

        assert f_minus[0] == test_cof, f"Expected f_minus[0] = {test_cof}, Actual: {f_minus[0]}"
        assert f_plus[0] == -test_cof, f"Expected f_plus[0] = {-test_cof}, Actual: {f_plus[0]}"


    def test_goal_state1(self):
        # Constants
        aps = AdaptivePusherSliderStickingForceInputSystem()
        theta = jnp.array([[0.0, aps.s_length/2.0]])

        # Create Goal State
        x_goal = aps.goal_state(theta)

        assert x_goal.shape == (1, 3)

        assert jnp.isclose(x_goal.at[0, 2].get(), 0.0), f"Expected x_goal[0, 2] = 0.0, received: {x_goal.at[0, 2]}"

    def test__f1(self):
        # Constants
        aps = AdaptivePusherSliderStickingForceInputSystem()
        x = jnp.array([[0.0, 0.0, 0.0]])
        theta = jnp.array([[0.0, aps.s_length/2.0]])

        # Compute f
        f = aps._f(x, theta)

        assert f.shape == (1, 3, 1), f"Expected f.shape = (1, 3, 1), received: {f.shape}"

        assert jnp.all(
            jnp.isclose(f, jnp.zeros((1, 3, 1)))
        ), f"Expected f = 0.0, received: {f.get()}"

    def test__F1(self):
        # Constants
        aps = AdaptivePusherSliderStickingForceInputSystem()
        x = jnp.array([[0.0, 0.0, 0.0]])
        theta = jnp.array([[0.0, aps.s_length/2.0]])

        # Compute f
        F = aps._F(x, theta)

        assert F.shape == (1, 3, 2), f"Expected F.shape = (1, 3, 2), received: {F.shape}"

        assert jnp.all(
            jnp.isclose(F, jnp.zeros((1, 3, 2)))
        ), f"Expected F = 0.0, received: {F.get()}"

    def test__g1(self):
        """
        test__g1
        Description:
            Tests the function _g for the Adaptive Pusher Slider.
        """
        # Constants
        aps = AdaptivePusherSliderStickingForceInputSystem()
        s_x = 0.0
        s_y = 1.0
        s_theta = jnp.pi/3.0
        x = jnp.array([[s_x, s_y, s_theta]])
        theta = jnp.array([[0.0, aps.s_length/2.0]])

        f_max, tau_max = aps.limit_surface_bounds()
        a = (1 / 10.0) * (1 / (f_max ** 2))
        b = (1 / 10.0) * (1 / (tau_max ** 2))

        # Compute g
        g = aps._g(x, theta)

        assert g.shape == (1, 3, 2), f"Expected F.shape = (1, 3, 2), received: {g.shape}"

        assert not jnp.all(
            jnp.isclose(g, jnp.zeros((1, 3, 2)))
        ), f"Expected g =/= 0.0, received: {g.get()}"

        assert jnp.isclose(
            g.at[0, AdaptivePusherSliderStickingForceInputSystem.S_THETA, AdaptivePusherSliderStickingForceInputSystem.F_X].get(),
            0.0,
        ), f"Expected g[0, {AdaptivePusherSliderStickingForceInputSystem.S_THETA}, {AdaptivePusherSliderStickingForceInputSystem.F_X}] = 0.0, received: {g.at[0, AdaptivePusherSliderStickingForceInputSystem.S_THETA, AdaptivePusherSliderStickingForceInputSystem.F_X].get()}"

        assert jnp.isclose(
            g.at[0, AdaptivePusherSliderStickingForceInputSystem.S_THETA, AdaptivePusherSliderStickingForceInputSystem.F_Y].get(),
            0.0,
        ), f"Expected g[0, {AdaptivePusherSliderStickingForceInputSystem.S_THETA}, {AdaptivePusherSliderStickingForceInputSystem.F_Y}] = 0.0, received: {g.at[0, AdaptivePusherSliderStickingForceInputSystem.S_THETA, AdaptivePusherSliderStickingForceInputSystem.F_Y].get()}"

        # Investigate nonzero elements
        assert jnp.isclose(
            g.at[0, AdaptivePusherSliderStickingForceInputSystem.S_X, AdaptivePusherSliderStickingForceInputSystem.F_X].get(),
            jnp.cos(s_theta) * a,
        ), f"Expected g[0, {AdaptivePusherSliderStickingForceInputSystem.S_X}, {AdaptivePusherSliderStickingForceInputSystem.F_X}] = {jnp.cos(s_theta) * a}, received: {g.at[0, AdaptivePusherSliderStickingForceInputSystem.S_X, AdaptivePusherSliderStickingForceInputSystem.F_X].get()}"
        assert jnp.isclose(
            g.at[0, AdaptivePusherSliderStickingForceInputSystem.S_X, AdaptivePusherSliderStickingForceInputSystem.F_Y].get(),
            -jnp.sin(s_theta) * a,
        ), f"Expected g[0, {AdaptivePusherSliderStickingForceInputSystem.S_X}, {AdaptivePusherSliderStickingForceInputSystem.F_Y}] = {-jnp.sin(s_theta) * a}, received: {g.at[0, AdaptivePusherSliderStickingForceInputSystem.S_X, AdaptivePusherSliderStickingForceInputSystem.F_Y].get()}"

        assert jnp.isclose(
            g.at[0, AdaptivePusherSliderStickingForceInputSystem.S_Y, AdaptivePusherSliderStickingForceInputSystem.F_X].get(),
            jnp.sin(s_theta) * a,
        ), f"Expected g[0, {AdaptivePusherSliderStickingForceInputSystem.S_Y}, {AdaptivePusherSliderStickingForceInputSystem.F_X}] = {jnp.sin(s_theta) * a}, received: {g.at[0, AdaptivePusherSliderStickingForceInputSystem.S_Y, AdaptivePusherSliderStickingForceInputSystem.F_X].get()}"

        assert jnp.isclose(
            g.at[0, AdaptivePusherSliderStickingForceInputSystem.S_Y, AdaptivePusherSliderStickingForceInputSystem.F_Y].get(),
            jnp.cos(s_theta) * a,
        ), f"Expected g[0, {AdaptivePusherSliderStickingForceInputSystem.S_Y}, {AdaptivePusherSliderStickingForceInputSystem.F_Y}] = {jnp.cos(s_theta) * a}, received: {g.at[0, AdaptivePusherSliderStickingForceInputSystem.S_Y, AdaptivePusherSliderStickingForceInputSystem.F_Y].get()}"

    def test__G1(self):
        # Constants
        aps = AdaptivePusherSliderStickingForceInputSystem()
        s_x = 0.0
        s_y = 1.0
        s_theta = jnp.pi / 3.0
        x = jnp.array([[s_x, s_y, s_theta]])
        theta = jnp.array([[0.0, aps.s_length / 2.0]])

        f_max, tau_max = aps.limit_surface_bounds()
        a = (1 / 10.0) * (1 / (f_max ** 2))
        b = (1 / 10.0) * (1 / (tau_max ** 2))

        # Compute g
        G = aps._G(x, theta)

        batch_size = x.shape[0]

        assert G.shape == (batch_size, aps.n_dims, aps.n_controls, aps.n_params),\
            f"Expected F.shape = ({batch_size}, {aps.n_dims}, {aps.n_controls}, {aps.n_params}), received: {G.shape}"

        # Check where zeros should be
        assert jnp.isclose(
            G.at[
                0,
                AdaptivePusherSliderStickingForceInputSystem.S_THETA,
                AdaptivePusherSliderStickingForceInputSystem.F_X,
                AdaptivePusherSliderStickingForceInputSystem.C_X,
            ].get(),
            0.0,
        ), f"Expected G[0, {AdaptivePusherSliderStickingForceInputSystem.S_THETA}, {AdaptivePusherSliderStickingForceInputSystem.F_X}, {AdaptivePusherSliderStickingForceInputSystem.C_X}] = 0.0, received: {G.at[0, AdaptivePusherSliderStickingForceInputSystem.S_THETA, AdaptivePusherSliderStickingForceInputSystem.F_X, AdaptivePusherSliderStickingForceInputSystem.C_X].get()}"

        assert jnp.isclose(
            G.at[
                0,
                AdaptivePusherSliderStickingForceInputSystem.S_THETA,
                AdaptivePusherSliderStickingForceInputSystem.F_Y,
                AdaptivePusherSliderStickingForceInputSystem.C_Y,
            ].get(),
            0.0,
        ), f"Expected G[0, {AdaptivePusherSliderStickingForceInputSystem.S_THETA}, {AdaptivePusherSliderStickingForceInputSystem.F_Y}, {AdaptivePusherSliderStickingForceInputSystem.C_Y}] = 0.0, received: {G.at[0, AdaptivePusherSliderStickingForceInputSystem.S_THETA, AdaptivePusherSliderStickingForceInputSystem.F_Y, AdaptivePusherSliderStickingForceInputSystem.C_Y].get()}"

        # Check where NON-zeros should be
        assert jnp.isclose(
            G.at[
                0,
                AdaptivePusherSliderStickingForceInputSystem.S_THETA,
                AdaptivePusherSliderStickingForceInputSystem.F_X,
                AdaptivePusherSliderStickingForceInputSystem.C_Y,
            ].get(),
            -b,
        ), f"Expected G[0, {AdaptivePusherSliderStickingForceInputSystem.S_THETA}, {AdaptivePusherSliderStickingForceInputSystem.F_X}, {AdaptivePusherSliderStickingForceInputSystem.C_Y}] = {-b}, received: {G.at[0, AdaptivePusherSliderStickingForceInputSystem.S_THETA, AdaptivePusherSliderStickingForceInputSystem.F_X, AdaptivePusherSliderStickingForceInputSystem.C_Y].get()}"

        assert jnp.isclose(
            G.at[
                0,
                AdaptivePusherSliderStickingForceInputSystem.S_THETA,
                AdaptivePusherSliderStickingForceInputSystem.F_Y,
                AdaptivePusherSliderStickingForceInputSystem.C_X,
            ].get(),
            b,
        ), f"Expected G[0, {AdaptivePusherSliderStickingForceInputSystem.S_THETA}, {AdaptivePusherSliderStickingForceInputSystem.F_Y}, {AdaptivePusherSliderStickingForceInputSystem.C_X}] = {b}, received: {G.at[0, AdaptivePusherSliderStickingForceInputSystem.S_THETA, AdaptivePusherSliderStickingForceInputSystem.F_Y, AdaptivePusherSliderStickingForceInputSystem.C_X].get()}"

if __name__ == '__main__':
    unittest.main()