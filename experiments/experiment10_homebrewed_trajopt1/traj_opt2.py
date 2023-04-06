"""
traj_opt2.py
Description:
    Running trajectory optimization and then observing how quickly trajectory optimization runs online.
"""

import sys, time, datetime

import matplotlib.pyplot as plt
import jax.numpy as jnp

import yaml

sys.path.append('../../')
#from src.python.pusher_slider import PusherSliderStickingVelocityInputSystem
from src.python.pusher_slider_sticking_force_input import PusherSliderStickingForceInputSystem

from src.python.simple_traj_opt import ic_traj_opt

# Constants
nominal_scenario = {
    "obstacle_center_x": 0.0,
    "obstacle_center_y": 0.0,
    "obstacle_radius": 0.2,
}
ps = PusherSliderStickingForceInputSystem(
    nominal_scenario=nominal_scenario,
)
# x0 = jnp.array([
#     [1.0], [1.0], [jnp.pi / 3], [0.0]
# ])
N_points = 50
dt = 0.1
u_step_size = 1.0

x0s = [
    jnp.array([-0.1, -0.1, jnp.pi / 3]),
    jnp.array([-0.1, -0.1, jnp.pi / 6]),
    jnp.array([-0.2, -0.1, jnp.pi / 6]),
    jnp.array([-0.2, -0.1, 0.0 / 3])
]

# [ [ -0.1, -0.1, pi/3, 0.02 ]' , [ -0.1, -0.1, pi/6, 0.02 ]' , [ -0.2, -0.1, pi/6, 0.02 ]' , [ -0.2, -0.1, 0, 0.02 ]' ]
opt_times = []
N_repetition = 1
traj_opt_results = {}
for target_index in range(len(x0s)):
    x0 = x0s[target_index]

    traj_opt_results["ic"+str(target_index)] = {}
    traj_opt_results["ic"+str(target_index)]["times"] = []
    for rep_index in range(N_repetition):
        u_opt, opt_time, final_loss = ps.ic_traj_opt(
            x0,
            horizon=N_points, num_traj_opt_iters=10, u_step_size=u_step_size,
        )
        traj_opt_results["ic" + str(target_index)]["times"].append(opt_time)
    traj_opt_results["ic"+str(target_index)]["example_u"] = u_opt

traj_opt_results["N_traj_points"] = N_points
traj_opt_results["dt"] = dt
traj_opt_results["u_step_size"] = u_step_size

print('Times taken for trajectory optimization:')
print(opt_times)
print(traj_opt_results)

# Run the optimized trajectory for one of the initial conditions.
u0 = u_opt[0, :].reshape((ps.n_controls,))
u_k = u0

x_k = x0
x = x0.T
traj_opt_results["simulation_times"] = []
for k in range(N_points):
    # print(ps.f(x_t,u_t))
    x_kp1 = x_k + ps.closed_loop_dynamics(x_k, u_k) * ps.dt
    # print("x_tp1 =",x_tp1)
    x = jnp.vstack(
        (x, x_kp1.T)
    )

    # Set variables for next step
    x_k = x_kp1

    start_compute_u_k = time.time()
    u_k = u_opt[k, :].reshape((2,))
    end_compute_u_k = time.time()

    traj_opt_results["simulation_times"].append( end_compute_u_k - start_compute_u_k )

plt.figure()
plt.plot(x[:,0],x[:,1])
plt.xlabel('s_x')
plt.ylabel('s_y')
plt.title('Trajectory With Constant u=[0.1;0] input')
plt.savefig("data/traj_opt2-simulate1-u_01-xy.png")

plt.figure()
plt.plot(jnp.arange(start=0,stop=N_points*dt+dt,step=dt),x[:,0])
plt.xlabel('t')
plt.ylabel('s_x')
plt.title('Trajectory With Constant u=[0.1;0] input')
plt.savefig("data/traj_opt2-simulate1-u_01-t_x.png")

plt.figure()
plt.plot(jnp.arange(start=0,stop=N_points*dt+dt,step=dt),x[:,1])
plt.xlabel('t')
plt.ylabel('s_y')
plt.title('Trajectory With Constant u=[0.1;0] input')
plt.savefig("data/traj_opt2-simulate1-u_01-t_y.png")

plt.figure()
plt.plot(jnp.arange(start=0,stop=N_points*dt+dt, step=dt),x[:,2])
plt.xlabel('t')
plt.ylabel('s_theta')
plt.savefig("data/traj_opt2-simulate1-u_01-t_sTheta.png")

plt.figure()
plt.plot(jnp.arange(start=0,stop=N_points*dt+dt,step=dt),x[:,3])
plt.xlabel('t')
plt.ylabel('p_y')
plt.savefig("data/traj_opt2-simulate1-u_01-t_p_y.png")

# Save data
filename = 'data/traj_opt2-traj_opt1_results-' + datetime.datetime.now().strftime("%B%d%Y-%I%M%p") + '.yml'

with open(filename, 'w') as outfile:
    yaml.dump(traj_opt_results, outfile, default_flow_style=False)

# Compare runtimes with clf-based control.

# read data from text file in clf_control1
# file1 = open('../data/clf_control1/clf_control1_results_20221128-045723.txt', 'r')
# Lines = file1.readlines()
# print(jnp.fromstring(Lines[0], sep=' '))
# setup_time_sos = jnp.fromstring(Lines[0],sep=' ')
# mean_data = []
# for line in Lines[1:]:
#     mean_data.append(jnp.mean(jnp.fromstring(line,sep=' ')) )
#
# X_axis = jnp.array([-1.0,1.0])
#
# fig_out = plt.figure()
# plt.bar(X_axis - 0.2, jnp.array([setup_time_sos[0], jnp.mean(jnp.array(mean_data) )]), 0.4, label = 'CLF')
# plt.bar(X_axis + 0.2, jnp.array([jnp.mean(jnp.array(traj_opt_results["ic0"]["times"])), jnp.mean(jnp.array(traj_opt_results["simulation_times"]))]), 0.4, label='TrajOpt')
# plt.xticks(X_axis, ["Precomputation", "Online Computation"])
# plt.ylabel('s')
# plt.legend()
# plt.title('Comparison of CLF and Trajectory Optimization (Static PS)')
#
# fig_out.savefig('data/traj_opt1/timing_comparison_static.png')
#
# # Save Two Figures Separately
# X_axis2 = jnp.array([0.0])
# fig_out = plt.figure()
# plt.bar(X_axis2 - 0.2, jnp.array([setup_time_sos[0]]), 0.4, label = 'CLF')
# plt.bar(X_axis2 + 0.2, jnp.array([ jnp.mean(jnp.array(traj_opt_results["ic0"]["times"])) ]), 0.4, label = 'TrajOpt')
# # plt.xticks(X_axis2, ["Precomputation"])
# plt.ylabel('s')
# plt.legend()
# plt.title('Precomputation Time of CLF vs. Trajectory Optimization Policies')
#
#
# fig_out.savefig('data/traj_opt1/timing_comparison_static-p1.png')
#
# fig_out = plt.figure()
# plt.bar(X_axis2 - 0.2, jnp.mean(jnp.array(mean_data)), 0.4, label='CLF')
# plt.bar(X_axis2 + 0.2, jnp.mean(jnp.array(traj_opt_results["simulation_times"])), 0.4, label='TrajOpt')
# # plt.xticks(X_axis2, ["Online Computation"])
# plt.ylabel('s')
# plt.legend()
# plt.title('Mean Time Per Iteration of CLF vs. Trajectory Optimization Policies')
#
#
# fig_out.savefig('data/traj_opt1/timing_comparison_static-p2.png')