"""
test_pusher_slider.jl
Description
    Tests the pusher-slider system and its associated functions.
"""

# Imports
using Plots
using Test
include("../../src/julia/slider_benchmarks.jl")

# ==============
# identify_mode

ps1 = GetPusherSlider()

# Test 1
x1 = [0.1; 0.1; pi/6; 0.0]
u1 = [0.2; -0.00]
@test identify_mode(ps1, x1, u1) == "Sticking"

# Test 2: Should be sliding up
u2 = [0.0; +0.2]
@test identify_mode(ps1, x1, u2) == "SlidingUp"

# Test 3: Should be sliding down
u3 = [0.05; -0.2]
@test identify_mode(ps1, x1, u3) == "SlidingDown"