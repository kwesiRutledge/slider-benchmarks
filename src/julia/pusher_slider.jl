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

mutable struct PusherSliderPlotParams
    showCenterOfMass::Bool
    sliderAlpha::Float64
    sliderColor
    pusherColor
    CoMColor
    MotionConeColor
    FrictionConeColor
    x_lims
    y_lims
    showMotionConeVectors::Bool
    showLineOfCentersOfRotation::Bool
    fPusher
end

function GetDefaultPusherSliderPlotParams()::PusherSliderPlotParams
    return PusherSliderPlotParams(
        false, # Show CoM
        0.5, # Slider Alpha
        :blue,:magenta,:red,
        :green,:orange,
        [-0.25, 0.75 ],[-0.25, 0.75 ],
        false,
        false,
        [])
end

function GetPusherSlider()::PusherSlider
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

"""
tau_max
Description
    Returns the maximum frictional force and amount of torque allowed by the limit surface.
"""
function limit_surface_bounds(ps::PusherSlider)
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

function show(ps::PusherSlider, x)
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
function show(ps::PusherSlider, x, pParams::PusherSliderPlotParams)
    # Constants
    #rectangle(w, h, x, y) = Shape(x + [0,w,w,0], y + [0,0,h,h]) # define a function that returns a Plots.Shape

    s_x = x[1]
    s_y = x[2]
    s_theta = x[3]
    p_y = x[4]

    p_x = -ps.s_length/2

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
    circle_center = [s_x; s_y] + rot * ( [ p_x; p_y ] + [ -ps.p_radius; 0] )
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
        plot!(
            [s_x],
            [s_y],
            c = :red,
            seriestype=:scatter,
        )
    end

    # Show Motion Cone vectors
    # ========================
    if pParams.showMotionConeVectors
        # prep
        circle_touch_point = [s_x; s_y] + rot * ( [ p_x ; p_y ] )
        rot2 = rotationMatrix(-(pi/2-s_theta))

        # Get Motion Cone Vectors, normalize them, and then scale them by box width
        v_plus, v_minus = get_motion_cone_vectors(ps,x)
        # println("v_plus = $(v_plus) while v_minus = $(v_minus)")

        v_plus = v_plus ./ (LinearAlgebra.norm(v_plus,2))
        v_plus = rot * v_plus .* (ps.s_length/2)

        v_minus = v_minus ./ (LinearAlgebra.norm(v_minus,2))
        v_minus = rot * v_minus .* (ps.s_length/2)

        # Plot
        v_stacked = hcat(v_plus,v_minus) 
        quiver!(
            circle_touch_point[1]*ones(2,1), circle_touch_point[2]*ones(2,1),
            quiver=(v_stacked[1,:],v_stacked[2,:]),
            c = pParams.MotionConeColor,
        )
    end

    # Show Line Of Centers of Rotation
    # ================================
    if pParams.showLineOfCentersOfRotation
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
get_motion_cone_constants
Description:
    Computes the motion cone vectors at the current state which determine when the
    pusher is sticking/sliding with the slider.
Inputs:
    x - Current state.
"""
function get_motion_cone_constants(ps::PusherSlider, x)
    # Constants
    g = 10
    f_max, m_max = limit_surface_bounds(ps)

    c = m_max / f_max

    mu_ps = ps.ps_cof

    p_x = -ps.s_length/2

    # Get Current state
    s_x = x[1]
    s_y = x[2]
    s_theta = x[3]
    p_y = x[4]

    # Compute vectors
    gamma_plus = (mu_ps*c.^2 - p_x * p_y + mu_ps*p_x^2)/( c.^2 + p_y.^2 - mu_ps * p_x * p_y )
    gamma_minus = (-mu_ps*c.^2 - p_x * p_y - mu_ps*p_x^2)/( c.^2 + p_y.^2 + mu_ps * p_x * p_y )

    return gamma_plus, gamma_minus
end

"""
get_motion_cone_vectors
Description:
    Computes the motion cone vectors at the current state which determine when the
    pusher is sticking/sliding with the slider.
Inputs:
    x - Current state.
"""
function get_motion_cone_vectors(ps::PusherSlider,x)
    # Constants
    gamma_plus, gamma_minus = get_motion_cone_constants(ps,x)

    # Algorithm
    return [1; gamma_plus], [1; gamma_minus]
end

"""
identify_mode
Description:
    Determines the mode of the sliding object w.r.t. the slider (mode is either sticking, sliding up, or sliding down).
    The mode is determined by the input u, a two-dimensional vector.
       u =  [ v_n ]
            [ v_t ]

"""
function identify_mode(ps::PusherSlider, x , u)::String
    # Constants

    # Variables
    v_n = u[1]
    v_t = u[2]

    # Motion Cone Vector Constants
    gamma_plus, gamma_minus = get_motion_cone_constants(ps,x)

    # Algorithm
    if ( v_t <= gamma_plus * v_n) && ( v_t >= gamma_minus * v_n )
        return "Sticking"
    elseif v_t > gamma_plus * v_n
        return "SlidingUp"
    else
        return "SlidingDown"
    end

end

function C(ps::PusherSlider, x)
    # Constants
    s_theta = x[3]

    # Algorithm
    return [    cos(s_theta)    sin(s_theta) ; 
                -sin(s_theta)   cos(s_theta) ]
    
end

function Q(ps, x)
    # Constants
    g = 10;
    f_max, m_max = limit_surface_bounds(ps)

    c = m_max / f_max

    p_x = -ps.s_length/2
    p_y = x[4]

    # Algorithm
    return (1/( c.^2 + p_x.^2 + p_y.^2 )) * 
            [   c.^2 + p_x.^2   p_x * p_y ; 
                p_x * p_y       c.^2 + p_y.^2 ]
end

"""
f1
Description:
    Dynamics of the "sticking" mode of the system.
"""
function f1( ps::PusherSlider, x , u )
    # Constants
    g = 10;
    f_max, m_max = limit_surface_bounds(ps)

    c = m_max / f_max

    p_x = -ps.s_length/2

    # State
    p_y = x[4]

    # Algorithm
    C0 = C(ps, x)
    Q0 = Q(ps, x)

    b1 = (1/(c.^2 + p_x.^2+p_y.^2)) * [ -p_y p_x ]
    c1 = [ 0 0 ]

    P1 = I

    dxdt = [    C0' * Q0 * P1 ;
                b1 ; 
                c1 ] * u

    return dxdt
end

"""
f2
Description:
    Dynamics of the "SlidingUp" mode.
"""
function f2( ps::PusherSlider, x , u )
    # Constants
    g = 10;
    f_max, m_max = limit_surface_bounds(ps)

    c = m_max / f_max

    p_x = -ps.s_length/2

    # Motion Cone Vector Constants
    gamma_plus, gamma_minus = get_motion_cone_constants(ps,x)

    # State
    p_y = x[4]

    # Algorithm
    C0 = C(ps, x)
    Q0 = Q(ps, x)

    b2 = (1/(c.^2 + p_x.^2+p_y.^2)) * [ -p_y+gamma_plus*p_x 0 ]
    c2 = [ -gamma_plus 0 ]

    P2 = [  1           0;
            gamma_plus  0 ]

    dxdt = [    C0' * Q0 * P2 ;
                b2 ;
                c2 ] * u
    
    return dxdt
end

"""
f3
Description:
    Dynamics of the "SlidingDown" mode.
"""
function f3( ps::PusherSlider, x , u )
    # Constants
    g = 10;
    f_max, m_max = limit_surface_bounds(ps)

    c = m_max / f_max

    p_x = -ps.s_length/2

    # Motion Cone Vector Constants
    gamma_plus, gamma_minus = get_motion_cone_constants(ps,x)

    # State
    p_y = x[4]

    # Algorithm
    C0 = C(ps, x)
    Q0 = Q(ps, x)

    b3 = (1/(c.^2 + p_x.^2+p_y.^2)) * [ -p_y+gamma_minus*p_x 0 ]
    c3 = [ -gamma_minus 0 ]

    P3 = [  1       0;
            gamma_minus 0 ]

    dxdt = [    C0' * Q0 * P3 ;
                b3 ; 
                c3 ] * u
    
    return dxdt
end

"""
f
Description:
    Defines the switched nonlinear dynamical system for the pusher slider system.
"""
function f( ps::PusherSlider , x , u )
    # Constants

    # Algorithm
    currMode = identify_mode(ps, x, u)
    if currMode == "Sticking"
        return f1(ps,x,u)
    elseif currMode == "SlidingUp"
        return f2(ps,x,u)
    elseif currMode == "SlidingDown"
        return f3(ps,x,u)
    else
        throw("There was an unexpected mode detected: " + currMode)
    end
end

"""
get_friction_cone_boundary_vectors
Description

Notes
    The frame of reference will be the same as that of Hogan and Rodriguez (2016).
    That is, into the block (orthogonal to surface) is the x-direction and along the surface is the y-direction.

    Vectors are returned in reference to this frame
"""
function get_friction_cone_boundary_vectors(ps::PusherSlider)
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
"""
function contact_point(ps::PusherSlider, x)
    # Constants
    s_x = x[1]
    s_y = x[2]
    s_theta = x[3]
    p_y = x[4]

    p_x = -ps.s_length/2

    rot = rotationMatrix(s_theta)

    # Algorithm
    return [s_x; s_y] + rot * [ p_x; p_y ]
end


"""
CoR_point_from_contact_force
Description:
    Computes the two line segments that define where the dual of the input
    contact force lies inside the slider.
"""
function CoR_point_from_contact_force(ps::PusherSlider, x , f)
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
    tilde_r_f = c / norm(r_f,2)
    tilde_r_f = tilde_r_f * r_f_direction # Has same direction as r_f

    println("tilde_r_f = $(tilde_r_f) with norm $(norm(tilde_r_f,2))")

    return r_f, tilde_r_f, x_bar
end
