"""
pusher_slider_sticking_fi.jl
Description:
    This file contains the code for modeling the pusher slider system
    with
    - sticking friction (so no movement of pusher w.r.t. slider) and
    - force inputs to the pusher slider.
"""

mutable struct PusherSliderStickingFI
    # Inertial Parameters
    s_mass::Float64
    # Geometric Parameters
    s_length::Float64
    s_width::Float64
    ps_cof::Float64
    st_cof::Float64
    p_radius::Float64
    # Center Of Mass Position
    p_x::Float64    # Center of Mass Position's x coordinate w.r.t. contact point
    p_y::Float64    # Center of Mass Position's y coordinate w.r.t. contact point
end

function GetPusherSliderStickingFI()::PusherSliderStickingFI
    # Default Parameters
    s_mass = 1.05 # kg
    s_length = 0.09 # m
    s_width = s_length
    ps_cof = 0.3
    st_cof = 0.35

    p_radius = 0.01 # m

    # Return object
    return PusherSliderStickingFI(
        s_mass,
        s_length, s_width,
        ps_cof, st_cof,
        p_radius,
        -s_length/2,0)
end

"""
tau_max
Description
    Returns the maximum frictional force and amount of torque allowed by the limit surface.
"""
function limit_surface_bounds(ps::PusherSliderStickingFI)
    # Constants
    g = 10

    # Create output
    f_max = ps.st_cof * ps.s_mass * g

    slider_area = ps.s_width*ps.s_length
    # circular_density_integral = 2*pi*((ps.s_length/2)^2)*(1/2)
    circular_density_integral = (1/12)*((ps.s_length/2).^2 + (ps.s_width/2).^2) * exp(1)
    
    tau_max = ps.st_cof * ps.s_mass * g * (1/slider_area) * circular_density_integral
    return f_max, tau_max
end

"""
check_state
Description:
    Checks if the state is valid.
"""
function check_state(ps::PusherSliderStickingFI, x)
    # Constants

    # Algorithm
    if length(x) != 3
        error("x must be a 3x1 vector")
    end

end

function show(ps::PusherSliderStickingFI, x)
    # Constants

    # Define Plot Params
    pParams = GetDefaultPusherSliderPlotParams()

    # Plot Rectangle
    return show(ps, x, pParams)
end

"""
Plot
Description:
"""
function show(ps::PusherSliderStickingFI, x, pParams::PusherSliderPlotParams)
    # Input Processing
    check_state(ps,x)
    
    # Constants
    #rectangle(w, h, x, y) = Shape(x + [0,w,w,0], y + [0,0,h,h]) # define a function that returns a Plots.Shape

    s_x = x[1]
    s_y = x[2]
    s_theta = x[3]
    
    p_y = ps.p_y
    p_x = ps.p_x

    # Create the Slider
    # =================

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
    r_n_t_corners = rotated_corners + [s_x*ones(1,4); s_y*ones(1,4)]

    p = plot(
            Shape(r_n_t_corners[1,:], r_n_t_corners[2,:]),
            fill=true,
            fillalpha=pParams.sliderAlpha,
            fillcolor=pParams.sliderColor, 
            xlims=pParams.x_lims, ylims=pParams.y_lims, 
            legend=false, aspect_ratio=:equal, size=(500,500)
        )

    # Create the Pusher
    # =================
    circle_center = contact_point(ps,x) + rot * ( [ -ps.p_radius; 0] )
    plot!(
        circleShape(circle_center[1],circle_center[2], ps.p_radius),
        seriestype=[:shape,],
        lw = 0.5,
        c = :blue,
        linecolor= :black,
    )

    # Draw Center Of Mass 
    # ===================
    if pParams.showCenterOfMass
        CoM = contact_point(ps,x) + rot * ( -[p_x; p_y])
        plot!(
            [CoM[1]],
            [CoM[2]],
            c = :red,
            seriestype=:scatter,
        )
    end

    # Show Motion Cone vectors
    # ========================
    if pParams.showMotionConeVectors
        # prep
        circle_touch_point = contact_point(ps,x)
        rot2 = rotationMatrix(-(pi/2-s_theta))

        # Get Motion Cone Vectors, normalize them, and then scale them by box width
        v_plus, v_minus = get_motion_cone_vectors(ps,x)
        # println("v_plus = $(v_plus) while v_minus = $(v_minus)")

        v_plus = v_plus ./ (LinearAlgebra.norm(v_plus,2))
        v_plus = rot2 * v_plus .* (ps.s_length/2)

        v_minus = v_minus ./ (LinearAlgebra.norm(v_minus,2))
        v_minus = rot2 * v_minus .* (ps.s_length/2)

        # Plot
        v_stacked = hcat(v_plus,v_minus) 
        quiver!(
            circle_touch_point[1]*ones(2,1), circle_touch_point[2]*ones(2,1),
            quiver=(v_stacked[1,:],v_stacked[2,:]),
            c = pParams.MotionConeColor,
        )
    end

    # Show Friction Cone vectors
    # ==========================
    if pParams.showFrictionConeVectors
        # Plot Friction Cone
        pusher_tip = contact_point(ps,x)
        rot2 = rotationMatrix(-(pi/2-s_theta))
        
        f_u, f_l = get_friction_cone_boundary_vectors(ps)
        # Scale unit vectors and rotate them into frame
        f_u = rot2 * f_u .* (ps.s_length/2)
        f_l = rot2 * f_l .* (ps.s_length/2)

        v_stacked = hcat( f_u, f_l )
        quiver!(
            pusher_tip[1]*ones(2,1),
            pusher_tip[2]*ones(2,1),
            c = pParams.FrictionConeColor,
            quiver=(
                v_stacked[1,:],
                v_stacked[2,:]
            ),
        )
    end

    # Show Line Of Centers of Rotation
    # ================================
    if pParams.showLineOfCentersOfRotation
        error("Not implemented yet")

        # Constants
        rot2 = rotationMatrix(-(pi/2-s_theta))

        # Get Motion Cone Vectors, normalize them, and then scale them by box width
        v_plus, v_minus = get_motion_cone_vectors(ps,x)
        println("v_plus: $(v_plus) with shape $(size(v_plus))")
        
        # Compute shortest distance between f_plus and f_minus from friction cone.
        f_u, f_l = get_friction_cone_boundary_vectors(ps)
        r_f_plus,tilde_r_f_plus,x_bar_plus = CoR_point_from_contact_force(
            ps,
            x,
            rot2 * f_u)
        r_f_minus,tilde_r_f_minus,x_bar_minus = CoR_point_from_contact_force(
            ps,
            x,
            rot2 * f_l)

        println("x_bar_plus = $(x_bar_plus) while x_bar_minus = $(x_bar_minus)")

        plot!(
            [x_bar_plus[1], x_bar_minus[1]],
            [x_bar_plus[2], x_bar_minus[2]],
            c = :black,
            lw = 2,
            seriestype=:scatter,
        )

        Z_plus = [s_x;s_y] + tilde_r_f_plus
        Z_minus = [s_x;s_y] + tilde_r_f_minus

        # Plot the End points of the line of centers of rotation
        plot!(
            [Z_plus[1],Z_minus[1]],
            [Z_plus[2],Z_minus[2]],
            c = pParams.FrictionConeColor,
            seriestype=:scatter,
        )
        v_stacked = hcat(tilde_r_f_plus, tilde_r_f_minus)
        quiver!(
            s_x*ones(2,1),
            s_y*ones(2,1),
            quiver=(v_stacked[1,:],v_stacked[2,:]),
            
        ) # Draws arrows from the center of mass to the end points of the line of centers of rotation
        
        # Plot Friction Cone
        pusher_tip = [s_x; s_y] + rot * ( [ p_x; p_y ] )
        
        f_u, f_l = get_friction_cone_boundary_vectors(ps)
        # Scale unit vectors and rotate them into frame
        f_u = rot2 * f_u .* (ps.s_length/2)
        f_l = rot2 * f_l .* (ps.s_length/2)

        v_stacked = hcat( f_u, f_l )
        quiver!(
            pusher_tip[1]*ones(2,1),
            pusher_tip[2]*ones(2,1),
            c = pParams.FrictionConeColor,
            quiver=(
                v_stacked[1,:],
                v_stacked[2,:]
            ),
        )

        # Plot line from CoM to Friction Cone
        v_stacked = hcat(-r_f_plus, -r_f_minus)
        quiver!(
            s_x*ones(2,1),
            s_y*ones(2,1),
            quiver=(
                v_stacked[1,:],
                v_stacked[2,:]
            ),
        )
    end

    # Return Plot Object
    return p
end

"""
Jp
Description:
    This jacobian matrix is created by using the formula in Zhou et al.'s 
    2020 paper on the pusher slider system. The formula is:
        Jp = [  1 0 -p_y;
                0 1 p_x]
"""
function Jp(ps::PusherSliderStickingFI)
    # Constants
    p_y = ps.p_y
    p_x = ps.p_x

    # Algorithm
    return [1 0 p_x; 0 1 -p_y] # The p_y and p_x are flipped due to the coordinate system choice
end

"""
get_motion_cone_constants
Description:
    Computes the motion cone vectors at the current state which determine when the
    pusher is sticking/sliding with the slider.
Inputs:
    x - Current state.
"""
function get_motion_cone_vectors(ps::PusherSliderStickingFI, x)
    # Input Processing
    check_state(ps,x)
    
    # Constants
    f_max, m_max = limit_surface_bounds(ps)

    a = (1/10) * (1/(f_max.^2))
    b = (1/10) * (1/(m_max.^2))

    A = diagm([a,a,b])

    # Compute vectors
    f_u, f_l = get_friction_cone_boundary_vectors(ps)

    u_u = Jp(ps) * A * Jp(ps)' * f_u
    u_l = Jp(ps) * A * Jp(ps)' * f_l

    return u_u, u_l
end

"""
dynamics
Description:
    Computes the dynamics of the pusher slider system with given state and
    force input to sticking pusher slider.
"""
function dynamics(ps::PusherSliderStickingFI, x, u)
    # Input Processing
    check_state(ps,x)

    # Constants
    theta_W = x[3]
    Rtheta = rotationMatrix(theta_W)

    f_max, m_max = limit_surface_bounds(ps)
    
    a = (1/10) * (1/(f_max.^2))
    b = (1/10) * (1/(m_max.^2))

    A = diagm([a,a,b])

    Jp_transpose = Jp(ps)'

    # Algorithm
    return Rtheta * A * Jp_transpose * u

end

"""
get_friction_cone_boundary_vectors
Description

Notes
    The frame of reference will be the same as that of Hogan and Rodriguez (2016).
    That is, into the block (orthogonal to surface) is the x-direction and along the surface is the y-direction.

    Vectors are returned in reference to this frame
"""
function get_friction_cone_boundary_vectors(ps::PusherSliderStickingFI)
    # Constants

    # Algorithm
    f_u = LinearAlgebra.nullspace([1 ps.ps_cof])
    f_u = f_u ./ LinearAlgebra.norm(f_u,2)

    f_l = LinearAlgebra.nullspace([1 -ps.ps_cof])
    f_l = f_l ./ LinearAlgebra.norm(f_l,2)

    return f_u, f_l
end

"""
contact_point
Description:
    Computes the contact point of the pusher with the slider.
    This point should always be in the center of the object.
"""
function contact_point(ps::PusherSliderStickingFI, x)
    # Constants
    s_x = x[1]
    s_y = x[2]
    s_theta = x[3]

    rot = rotationMatrix(s_theta)

    # Algorithm
    return [s_x; s_y] + rot * [ -(ps.s_length/2) ; 0 ]
end


"""
CoR_point_from_contact_force
Description:
    Computes the two line segments that define where the dual of the input
    contact force lies inside the slider.
"""
function CoR_point_from_contact_force(ps::PusherSliderStickingFI, x , f)
    # Constants
    s_x = x[1]
    s_y = x[2]
    s_theta = x[3]
    p_y = x[4]
    p_x = -ps.s_length/2

    f_max, m_max = limit_surface_bounds(ps)

    # f_u, f_l = get_friction_cone_boundary_vectors(ps)

    # Algorithm

    x_bar = contact_point(ps, x) + f * ((([s_x; s_y]-contact_point(ps, x))'*f)/(f'*f)) 
    
    # Compute r_f
    r_f_direction = [s_x;s_y] - x_bar
    r_f_direction = r_f_direction ./ norm(r_f_direction,2)

    rot_f = rotationMatrix(-s_theta) * f
    r_f = abs(p_x * (rot_f[2])/(norm(rot_f,2)) ) * r_f_direction

    # Compute tilde_r_f
    c = m_max / f_max
    tilde_r_f = (c^2) / norm(r_f,2)
    tilde_r_f = tilde_r_f * r_f_direction # Has same direction as r_f

    println("tilde_r_f = $(tilde_r_f) with norm $(norm(tilde_r_f,2))")

    return r_f, tilde_r_f, x_bar
end
