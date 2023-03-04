"""
base_sim.jl
Description:
    Differential flatness experiment 7
    Should verify whether or not our system's output really is differentially flat.
"""

# Imports
using Plots
include("../../src/julia/slider_benchmarks.jl")

# Plot Centers of Rotation
x0 = [0.1; 0.1; pi/6; 0.00]
ps = GetPusherSlider()
pParams = GetDefaultPusherSliderPlotParams()
pParams.showCenterOfMass = true
pParams.showMotionConeVectors = true
pParams.showLineOfCentersOfRotation = true
pParams.x_lims = [0.0,0.25]
pParams.y_lims = [0.0,0.25]

p = show(ps, x0, pParams)

savefig(p, "ps_with_cor_line.png")

x2 = [0.1; 0.1; pi/6; 0.01]
p2 = show(ps, x2, pParams)
savefig(p2, "ps_with_cor_line2.png")

x3 = [0.1; 0.1; pi/6; 0.02]
p3 = show(ps, x3, pParams)
savefig(p3, "ps_with_cor_line3.png")
