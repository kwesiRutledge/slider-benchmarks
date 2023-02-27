"""
base_sim.jl
Description:
    This script quickly constructs the Pusher-Slider's figure and then draws the
    friction cone vectors on top of it.
"""

using Plots

include("../../src/julia/slider_benchmarks.jl")

# Create the figure
x = range(0, 10, length=100)
y = sin.(x)
p1 = plot(x, y)
savefig(p1, "base_sim1.png")

# Create Pusher Slider
ps = GetPusherSlider()
psPParams = GetDefaultPusherSliderPlotParams()
psPParams.showCenterOfMass = true

p2 = show(ps, [0.1; 0.1; pi/6; 0.02], psPParams)
savefig(p2, "ps_plot1.png")

# Create Motion Cone vector plot
x2 = [0.1; 0.1; pi/6; 0.02]
psPParams = GetDefaultPusherSliderPlotParams()
psPParams.showMotionConeVectors = true
psPParams.x_lims = [0.0,0.25]
psPParams.y_lims = [0.0,0.25]

p3 = show(ps, x2, psPParams)

savefig(p3, "ps_with_mc1.png")
