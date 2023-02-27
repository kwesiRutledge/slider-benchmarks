"""
base_sim.jl
Description:
    The basic script used to visualize our Pusher-Slider.
"""

include("../../src/julia/slider_benchmarks.jl")

# Constants
ps = GetPusherSlider()

# Simulate a trajectory
x0 = [0.1; 0.1; pi/6; 0.02]

u0 = [0.2; -0.05]

println(f(ps,x0,u0))

anim = @animate for k in 1:50
    pParams = GetDefaultPusherSliderPlotParams()
    p1 = show(ps,x0,pParams)
end

gif(anim, "test_gif1.gif",fps=50)

function animate1()
    # Create animation using simulation
    N = 200
    dt = 0.01
    x_k = x0

    x_h = x_k
    for k in 0:N-1
        x_kp1 = x_k + dt * f(ps,x_k,u0)

        # Update History
        x_h = hcat(x_h,x_kp1)

        x_k = copy(x_kp1)
    end

    anim2 = @animate for k in 0:N
        pParams = GetDefaultPusherSliderPlotParams()
        pParams.showMotionConeVectors = true
        p2 = show(ps, x_h[:,k+1], pParams)
    end

    Plots.apng(anim2, "test_apng1.png",fps=Int(1/dt))
end

animate1()