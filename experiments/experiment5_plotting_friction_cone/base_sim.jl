"""
base_sim.jl
Description:
    This script quickly constructs the Pusher-Slider's figure and then draws the
    friction cone vectors on top of it.
"""

using Plots

include("../../src/julia/pusher_slider.jl")

# Create the figure
x = range(0, 10, length=100)
y = sin.(x)
p1 = plot(x, y)
savefig(p1, "base_sim1.png")

# Create Pusher Slider
ps = GetPusherSlider()

p2 = Plot(ps, [0.1; 0.1; pi/6; 0.02])
savefig(p2, "ps_plot1.png")
