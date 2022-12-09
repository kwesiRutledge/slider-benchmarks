"""
simulate_slider.py
Description:
    This file will simply plot trajectories of the slider over time
    (maybe I will also save videos/gifs?).
"""

import sys, time

import matplotlib.pyplot as plt
from jax import jit
import jax.numpy as jnp

sys.path.append('../')
from src.pusher_slider import PusherSliderSystem

# Constants
ps = PusherSliderSystem()
# compiled_f = jit(ps.f)

# Plot Trajectory When u = 0 for all time
x0 = ps.x()
dt = 0.1
t_stop = 10.0-dt

x_t = x0
x = x0.T
for t in jnp.arange(start=0,stop=t_stop,step=dt):
    u_t = jnp.array([[0],[0]])
    x_tp1 = x_t + ps.f(x_t,u_t) * dt
    x = jnp.vstack(
        (x,x_tp1.T)
    )

    # Set variables for next step
    x_t = x_tp1

plt.figure()
plt.plot(x[:,0],x[:,1])
plt.xlabel('s_x')
plt.ylabel('s_y')
plt.savefig("../images/example_simulate/simulate1-u_0.png")

# Plot Trajectory When u = [0.1;0] for all time
ps = PusherSliderSystem()
ps.p_y = 0.0

x0 = ps.x()
dt = 0.1
t_stop = 10.0-dt

u0 = jnp.array([[0.05],[0.0]])

x_t = x0
x = x0.T
for t in jnp.arange(start=0,stop=t_stop,step=dt):
    u_t = u0
    f = jnp.zeros((4,1))
    # print(ps.f(x_t,u_t))
    x_tp1 = x_t + ps.f(x_t,u_t) * dt
    # print("x_tp1 =",x_tp1)
    x = jnp.vstack(
        (x,x_tp1.T)
    )

    # Set variables for next step
    x_t = x_tp1

plt.figure()
plt.plot(x[:,0],x[:,1])
plt.xlabel('s_x')
plt.ylabel('s_y')
plt.title('Trajectory With Constant u=[0.1;0] input')
plt.savefig("../images/example_simulate/simulate1-u_01-xy.png")

plt.figure()
plt.plot(jnp.arange(start=0,stop=t_stop+dt,step=dt),x[:,0])
plt.xlabel('t')
plt.ylabel('s_x')
plt.title('Trajectory With Constant u=[0.1;0] input')
plt.savefig("../images/example_simulate/simulate1-u_01-t_x.png")

plt.figure()
plt.plot(jnp.arange(start=0,stop=t_stop+dt,step=dt),x[:,1])
plt.xlabel('t')
plt.ylabel('s_y')
plt.title('Trajectory With Constant u=[0.1;0] input')
plt.savefig("../images/example_simulate/simulate1-u_01-t_y.png")

plt.figure()
plt.plot(jnp.arange(start=0,stop=t_stop+dt,step=dt),x[:,2])
plt.xlabel('t')
plt.ylabel('s_theta')
plt.savefig("../images/example_simulate/simulate1-u_01-t_sTheta.png")

plt.figure()
plt.plot(jnp.arange(start=0,stop=t_stop+dt,step=dt),x[:,3])
plt.xlabel('t')
plt.ylabel('p_y')
plt.savefig("../images/example_simulate/simulate1-u_01-t_p_y.png")