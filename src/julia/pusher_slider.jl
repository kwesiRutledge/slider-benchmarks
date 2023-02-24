"""
pusher_slider.jl
"""

struct PusherSlider
    # Inertial Parameters
    s_mass::Float64
    # Geometric Parameters
    s_length::Float64
    s_width::Float64
    ps_cof::Float64
    st_cof::Float64
    p_radius::Float64
end

function GetPusherSlider()
    # Default Parameters
    s_mass = 1.05 # kg
    s_length = 0.09 # m
    s_width = s_length
    ps_cof = 0.3
    st_cof = 0.35

    p_radius = 0.01 # m

    # Return object
    return PusherSlider(s_mass, s_length, s_width, ps_cof, st_cof, p_radius)
end

function Plot(ps::PusherSlider, x)
    # Constants

    # Default x_lims and y_lims
    x_lims = [-0.25, 0.75 ]
    y_lims = [-0.25, 0.75 ]

    # Plot Rectangle
    return Plot(ps, x, x_lims, y_lims)
end

"""
Plot
Description:
"""
function Plot(ps::PusherSlider, x, x_lims, y_lims)
    # Constants
    #rectangle(w, h, x, y) = Shape(x + [0,w,w,0], y + [0,0,h,h]) # define a function that returns a Plots.Shape

    s_theta = x[3]

    # Plot Rectangle
    x_lb = -ps.s_width/2
    x_ub =  ps.s_width/2
    y_lb = -ps.s_length/2
    y_ub =  ps.s_length/2

    corners = [ x_lb x_lb x_ub x_ub;
                y_lb y_ub y_ub y_lb ]

    # Rotate and translate corners
    rot = [ cos(s_theta) -sin(s_theta);
            sin(s_theta)  cos(s_theta) ]

    rotated_corners = rot * corners
    r_n_t_corners = rotated_corners + [x[1]*ones(1,4); x[2]*ones(1,4)]

    p = plot(Shape(r_n_t_corners[1,:], r_n_t_corners[2,:]), fill=true, fillalpha=0.5, fillcolor=:blue, xlims=x_lims, ylims=y_lims, legend=false, aspect_ratio=:equal, size=(500,500))

    # Return Plot Object
    return p
end