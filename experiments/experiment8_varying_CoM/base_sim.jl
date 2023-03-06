"""
base_sim.jl
Description:
    Creating a base simulation for experiment 8.
    I just want to visualize two pusher sliders with different CoMs.
"""

# Imports
using Plots
include("../../src/julia/slider_benchmarks.jl")

# Constants
mu_ps = 0.4

# Define ps1
ps1 = GetPusherSliderStickingFI()
ps1.p_y = 0.01
ps1.ps_cof = mu_ps

x0 = [0.1; 0.1; pi/6]
pParams = GetDefaultPusherSliderPlotParams()
pParams.showCenterOfMass = true
pParams.x_lims = [0, 0.25]
pParams.y_lims = [0, 0.25]
pParams.showFrictionConeVectors = true
pParams.showMotionConeVectors = true

p1 = show(ps1, x0, pParams)
savefig(p1, "ps1_py_0_01.png")

# Define ps2

ps2 = GetPusherSliderStickingFI()
ps2.p_y = 0.03
ps2.ps_cof = mu_ps

p2 = show(ps2, x0, pParams)
savefig(p2, "ps2_py_0_05.png")