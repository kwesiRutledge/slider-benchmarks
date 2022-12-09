classdef PusherSliderUnknownCoM < handle
    %PusherSliderUnknownCoM A representation of the Pusher-Slider System with Unknown Center of Mass
    %   A reinterpretation of the model from `Feedback Control of the 
    %   Pusher-Slider System: A Story of Hybrid and Underactuated
    %   Contact Dynamics` by Francois Robert Hogan and Alberto Rodriguez.
    %   https://arxiv.org/abs/1611.08268
    %   We add the complication that the center of mass is not necessarily in the geometric center of
    %   the sliding object.
    %
    %Notes:
    %   There is a quasi-static assumption in this system. i.e. "The quasi-static assumption suggests
    %   that at low velocities, frictional contact forces dominate and inertial forces do not have
    %   a decisive role in determining the motion of the slider."
    
    properties
        % State of the System
        s_x;        %x coordinate for the CoM of the Slider
        s_y;        %y coordinate for the CoM of the Slider
        s_theta;    %orientation of the CoM of the Slider w.r.t. Frame of reference a
        p_x;        %x position of the pusher w.r.t. Frame of reference b (attached to slider)
        p_y;        %y position of the pusher w.r.t. Frame of reference b (attached to slider)
        % Unknown Parameter
        p_y_offset; %offset describing how far and in which direction the CoM of the slider is 
                    %from the geometric center
        % Input of the System
        v_n;        %speed of the pusher in the direction normal to the edge of the sliding block
        v_t;        %speed of the pusher in the direction tangent to the edge of the sliding block
        % Physical Parameters of System
        s_length;   %length of the slider
        s_width;    %width of the slider
        s_mass;     %mass of the slider
        ps_cof;     %Coefficient of friction between the pusher and the slider
        st_cof;     %Coefficient of friction between the slider and the table
        p_radius;   %Radius of the pusher
    end
    
    methods
        function ps = PusherSliderUnknownCoM()
            %PusherSlider Construct an instance of this class
            %   Detailed explanation goes here
            
            %Define Default Physical Parameters
            ps.s_width = 0.09;
            ps.s_length = 0.09;
            ps.s_mass = 1.05; %kg
            ps.ps_cof = 0.3;
            ps.st_cof = 0.35;

            ps.p_radius = 0.01;
            
            % Define Initial State
            ps.s_x = 0.1;
            ps.s_y = 0.1;
            ps.s_theta = pi/6;
            ps.p_x = ps.s_width/2;
            ps.p_y = 0.02;

            % Define Initial Input
            ps.v_n = 0.01;
            ps.v_t = 0.03;

            % Define Offset Value
            ps.p_y_offset = 0; % Choose zero by default.
            
            
        end
        
        function [plot_handles,circ1] = show(varargin)
            %show Shows the system in a figure window.
            %   Uses the provided physical parameters to show the slider
            %   and pusher.
            
            % Input Processing
            % ================
            
            ps = varargin{1};
            
            % Constants
            % =========

            lw = 2.0;
            sliderColorChoice = 'blue';
            pusherColorChoice = 'magenta';


            % Creating Slider
            % ===============
            
            %Create lines.
            x_lb = -ps.s_width/2;
            x_ub =  ps.s_width/2;
            y_lb = -ps.s_length/2;
            y_ub =  ps.s_length/2;
            
            corners = [ x_lb, x_lb, x_ub, x_ub ;
                        y_lb, y_ub, y_ub, y_lb ];
            
            %Rotate and translate this.
            rot = [ cos(ps.s_theta), -sin(ps.s_theta) ;
                    sin(ps.s_theta), cos(ps.s_theta) ];
                
            rotated_corners = rot * corners;
            
            r_n_t_corners = rotated_corners + [ ps.s_x * ones(1,4); ps.s_y * ones(1,4) ]; %Rotated and translated corners
            
            % Plot
            hold on;
            for corner_idx = 1:size(r_n_t_corners,2)-1
                plot_handles(corner_idx) = plot( ...
                    r_n_t_corners(1,corner_idx+[0:1]) , ...
                    r_n_t_corners(2,corner_idx+[0:1]) , ...
                    'LineWidth', lw , ...
                    'Color', sliderColorChoice ...
                );
            end
            plot_handles(length(plot_handles)+1) = plot( ...
                    r_n_t_corners(1,[4,1]) , ...
                    r_n_t_corners(2,[4,1]) , ...
                    'LineWidth', lw , ...
                    'Color', sliderColorChoice ...
                );
            hold off;

            % Create Pusher
            % =============

            % Create the circle
            circle_center = [ ps.s_x ; ps.s_y ] + rot*([ -ps.p_x ; ps.p_y ] + [ -ps.p_radius ; 0 ]);
            circle_ll = circle_center - ps.p_radius *ones(2,1);

            hold on;
            circ1 = rectangle(  'Position', [circle_ll(1),circle_ll(2),2*ps.p_radius*ones(1,2)],...
                                'Curvature',[1,1] , ...
                                'LineWidth', lw , ...
                                'FaceColor', pusherColorChoice );
            hold off;

            
        end

        function [ gamma_t , gamma_b ] = get_motion_cone_vectors(ps)
            %get_motion_cone_vectors
            %Description:
            %
            %
            %Usage:
            %     [gamma_t, gamma_b] = ps.get_motion_cone_vectors();
            %
            %Questions:
            %   1. What is the coefficient mu in this formula? (Which one is it corresponding to?)
            %   2. How do you normally compute the integral in m_max?

            %Constants
            g = 10;
            f_max = ps.st_cof * ps.s_mass*g;
            m_max = ps.st_cof * ps.s_mass*g * (ps.s_width/2); % The last term is meant to come from a sort of mass distribution/moment calculation. ???
            c = f_max / m_max;
            mu = ps.st_cof; %Which coefficient of friction is this supposed to be?

            p_y_tilde = ps.p_y + ps.p_y_offset;

            gamma_t = (mu*c.^2 - ps.p_x * p_y_tilde + mu*ps.p_x^2)/( c.^2 + p_y_tilde.^2-mu*ps.p_x*p_y_tilde );

            gamma_b = (-mu*c.^2 - ps.p_x * p_y_tilde - mu*ps.p_x^2)/( c.^2 + p_y_tilde.^2 + mu*ps.p_x*p_y_tilde );

        end

        function [ mode_name ] = identify_mode( ps , u )
            %identify_mode
            %Description:
            %   Determines the mode of the sliding object w.r.t. the slider (mode is either sticking, sliding up, or sliding down).
            %   The mode is determined by the input u, a two-dimensional vector.
            %       u = [ v_n ]
            %           [ v_t ]

            % Constants

            % Variables

            v_n = u(1);
            v_t = u(2);

            [gamma_t, gamma_b] = ps.get_motion_cone_vectors();

            % Algorithm
            if ( v_t <= gamma_t * v_n ) && ( v_t >= gamma_b * v_n )
                mode_name = 'Sticking';
            elseif v_t > gamma_t*v_n
                mode_name = 'SlidingUp';
            else
                mode_name = 'SlidingDown';
            end

        end


        function [ C_out ] = C(ps)
            %C
            %Description:
            %   Creates the rotation matrix used in the pusher slider system's dynamics.
            %   Note: This is NOT the C matrix from the common linear system's y = Cx + v.


            % Constants
            theta = ps.s_theta;

            % Algorithm

            C_out = [ cos(theta) , sin(theta) ; -sin(theta) , cos(theta) ];

        end

        function [ Q_out ] = Q(ps)
            %Q
            %Description:
            %   Creates the Q matrix defined in the equations of motion.

            % Constants
            g = 10;
            f_max = ps.st_cof * ps.s_mass*g;
            m_max = ps.st_cof * ps.s_mass*g * (ps.s_width/2); % The last term is meant to come from a sort of mass distribution/moment calculation. ???
            c = f_max / m_max;
            p_x = ps.p_x;
            p_y_tilde = ps.p_y + ps.p_y_offset;

            % Algorithm

            Q_out = (1/( c.^2 + p_x.^2 + p_y_tilde.^2 )) * ...
                [ c.^2 + p_x.^2, p_x * p_y_tilde ; p_x * p_y_tilde, c.^2 + p_y_tilde.^2 ];

        end

        function set_state(ps,x)
            %set_state
            %Description:
            %   Sets the state of the pusher slider according to the state x
            %   where
            %           [   s_x   ]
            %       x = [   s_y   ]
            %           [ s_theta ]
            %           [   p_y   ]

            % Algorithm

            ps.s_x = x(1);
            ps.s_y = x(2);
            ps.s_theta = x(3);
            ps.p_y = x(4);

        end

        function set_input(ps,u)
            %set_input
            %Description:
            %   Gets the current input, where the input of the pusher slider x is
            %
            %       u = [ v_n ]
            %           [ v_t ]
            %Usage:
            %   ps.get_input()

            ps.v_n = u(1);
            ps.v_t = u(2);

        end

        function x = get_state(ps)
            %get_state
            %Description:
            %   Gets the state of the pusher slider according to the state x
            %   where
            %           [   s_x   ]
            %       x = [   s_y   ]
            %           [ s_theta ]
            %           [   p_y   ]
            %
            %Usage:
            %   ps.get_state()

            % Algorithm

            x = [ ps.s_x , ps.s_y , ps.s_theta, ps.p_y ]';

        end

        function u = get_input(ps)
            %get_input
            %Description:
            %   Gets the current input, where the input of the pusher slider x is
            %
            %       u = [ v_n ]
            %           [ v_t ]
            %Usage:
            %   ps.get_input()

            u = [ ps.v_n ; ps.v_t ];

        end

        function x_out = x(ps)
            %x
            %Description:
            %   
            %Usage:
            %   x = ps.x()

            x_out = ps.get_state();
        end

        function u_out = u(ps)
            %u
            %Description:
            %   Returns the current input of the pusher slider system.

            u_out = ps.get_input();

        end


        function [ dxdt ] = f1( ps , x , u )
            %f1
            %Description:
            %   Continuous dynamics of the sticking mode of contact between pusher and slider.

            % Constants
            ps.set_state(x);
            C0 = ps.C();
            Q0 = ps.Q();

            g = 10;
            f_max = ps.st_cof * ps.s_mass*g;
            m_max = ps.st_cof * ps.s_mass*g * (ps.s_width/2); % The last term is meant to come from a sort of mass distribution/moment calculation. ???
            c = m_max / f_max;

            p_x = ps.p_x;
            p_y_tilde = ps.p_y + ps.p_y_offset;

            % Algorithm
            b1 = [ -p_y_tilde/(c.^2 + p_x.^2+p_y_tilde.^2) , p_x ];

            c1 = [ 0 , 0 ];

            P1 = eye(2);

            dxdt = [    C0' * Q0 * P1 ; ...
                        b1 ; ...
                        c1 ] * u;

        end

        function [ dxdt ] = f2( ps , x , u )
            %f2
            %Description:
            %   Continuous dynamics of the SlidingUp mode of contact between pusher and slider.

            % Constants
            ps.set_state(x);
            C0 = ps.C();
            Q0 = ps.Q();

            g = 10;
            f_max = ps.st_cof * ps.s_mass*g;
            m_max = ps.st_cof * ps.s_mass*g * (ps.s_width/2); % The last term is meant to come from a sort of mass distribution/moment calculation. ???
            c = m_max / f_max;

            p_x = ps.p_x;
            p_y_tilde = ps.p_y + ps.p_y_offset;

            [gamma_t,gamma_b] = ps.get_motion_cone_vectors();

            % Algorithm
            b2 = [ (-p_y_tilde+gamma_t*p_x)/(c.^2 + p_x.^2+p_y_tilde.^2) , 0 ];

            c2 = [ -gamma_t , 0 ];

            P2 = [1, 0; gamma_t, 0];

            dxdt = [    C0' * Q0 * P2 ; ...
                        b2 ; ...
                        c2 ] * u;

        end

        function [ dxdt ] = f3( ps , x , u )
            %f3
            %Description:
            %   Continuous dynamics of the SlidingDown mode of contact between pusher and slider.

            % Constants
            ps.set_state(x);
            C0 = ps.C();
            Q0 = ps.Q();

            g = 10;
            f_max = ps.st_cof * ps.s_mass*g;
            m_max = ps.st_cof * ps.s_mass*g * (ps.s_width/2); % The last term is meant to come from a sort of mass distribution/moment calculation. ???
            c = m_max / f_max;

            p_x = ps.p_x;
            p_y_tilde = ps.p_y + ps.p_y_offset;

            [gamma_t,gamma_b] = ps.get_motion_cone_vectors();

            % Algorithm
            b3 = [ (-p_y_tilde+gamma_b*p_x)/(c.^2 + p_x.^2+p_y_tilde.^2) , 0 ];

            c3 = [ -gamma_b , 0 ];

            P3 = [1, 0; gamma_b, 0];

            dxdt = [    C0' * Q0 * P3 ; ...
                        b3 ; ...
                        c3 ] * u;

        end

        function [ dxdt ] = f( ps , x , u )
            %f
            %Description:
            %   Defines a switched nonlinear dynamical system.

            % Algorithm
            %ps.set_state(x);

            x_init = ps.x();

            currMode = ps.identify_mode(u);

            switch currMode
                case 'Sticking'
                    dxdt = ps.f1(x,u);
                case 'SlidingUp'
                    dxdt = ps.f2(x,u);
                case 'SlidingDown'
                    dxdt = ps.f3(x,u);
                otherwise
                    error('There was a problem with identifying the current mode!')
            end

            % Set state to be what it was originally given as.
            ps.set_state(x_init);

        end

        function [ A , B ] = LinearizedSystemAbout(ps,x,u)
            %LinearizedSystemAbout
            %Description:
            %
            %Usage:
            %   [ A , B ] = ps.LinearizedSystemAbout(x,u)
            %

            % Constants

            x_init = ps.x();
            u_init = ps.u();

            % Algorithm

            ps.set_state(x);

            currMode = ps.identify_mode(u);

            syms s_x s_y s_theta p_y v_n v_t real;

            % Identify A
            symvar_x = [s_x; s_y; s_theta; p_y];
            symvar_u = [v_n; v_t];

            switch currMode
                case 'Sticking'
                    dxdt = ps.f1(symvar_x,symvar_u);
                case 'SlidingUp'
                    dxdt = ps.f2(symvar_x,symvar_u);
                case 'SlidingDown'
                    dxdt = ps.f3(symvar_x,symvar_u);
                otherwise
                    error('There was a problem with identifying the current mode!')
            end

            dfdx = jacobian(dxdt,symvar_x);
            dfdu = jacobian(dxdt,symvar_u);

            % Doesn't work.
            % symvar_x = x;
            % symvar_u = u;
            s_x = x(1); s_y = x(2); s_theta = x(3); p_y = x(4);
            v_n = u(1); v_t = u(2);

            A = eval( subs(dfdx) );
            B = eval( subs(dfdu) );

            % Reset to initial state and input
            ps.set_state(x_init);
            ps.set_input(u_init);

        end

        function [ A_out ] = A1(ps,x,u)
            %A1
            %Description:
            %   Computes the linearized system matrix A assuming that the motion cone is the 'sticking one'
            %Usage:
            %   [ A ] = ps.A1(x,u)
            %

            % Constants

            x_init = ps.x();
            u_init = ps.u();

            % Algorithm

            ps.set_state(x);

            currMode = ps.identify_mode(u);

            syms s_x s_y s_theta p_y v_n v_t real;

            % Identify A
            symvar_x = [s_x; s_y; s_theta; p_y];
            symvar_u = [v_n; v_t];

            % (Mode is assumed to be 1 (sticking))
            % Compute Jacobian
            dxdt = ps.f1(symvar_x,symvar_u);

            dfdx = jacobian(dxdt,symvar_x);
            dfdu = jacobian(dxdt,symvar_u);

            s_x = x(1); s_y = x(2); s_theta = x(3); p_y = x(4);
            v_n = u(1); v_t = u(2);

            A_out = eval( subs(dfdx) );

            % Reset to initial state and input
            ps.set_state(x_init);
            ps.set_input(u_init);

        end

        function [ A_out ] = A2(ps,x,u)
            %A2
            %Description:
            %   Computes the linearized system matrix A assuming that the motion cone is the 'sliding up' one
            %Usage:
            %   [ A ] = ps.A2(x,u)
            %

            % Constants

            x_init = ps.x();
            u_init = ps.u();

            % Algorithm

            ps.set_state(x);

            currMode = ps.identify_mode(u);

            syms s_x s_y s_theta p_y v_n v_t real;

            % Identify A
            symvar_x = [s_x; s_y; s_theta; p_y];
            symvar_u = [v_n; v_t];

            % (Mode is assumed to be 1 (sticking))
            % Compute Jacobian
            dxdt = ps.f2(symvar_x,symvar_u);

            dfdx = jacobian(dxdt,symvar_x);
            dfdu = jacobian(dxdt,symvar_u);

            s_x = x(1); s_y = x(2); s_theta = x(3); p_y = x(4);
            v_n = u(1); v_t = u(2);

            A_out = eval( subs(dfdx) );

            % Reset to initial state and input
            ps.set_state(x_init);
            ps.set_input(u_init);

        end

        function [ A_out ] = A3(ps,x,u)
            %A3
            %Description:
            %   Computes the linearized system matrix A assuming that the motion cone is the 
            %   'sliding down' one
            %Usage:
            %   [ A ] = ps.A3(x,u)
            %

            % Constants

            x_init = ps.x();
            u_init = ps.u();

            % Algorithm

            ps.set_state(x);

            currMode = ps.identify_mode(u);

            syms s_x s_y s_theta p_y v_n v_t real;

            % Identify A
            symvar_x = [s_x; s_y; s_theta; p_y];
            symvar_u = [v_n; v_t];

            % (Mode is assumed to be 1 (sticking))
            % Compute Jacobian
            dxdt = ps.f3(symvar_x,symvar_u);

            dfdx = jacobian(dxdt,symvar_x);
            dfdu = jacobian(dxdt,symvar_u);

            s_x = x(1); s_y = x(2); s_theta = x(3); p_y = x(4);
            v_n = u(1); v_t = u(2);

            A_out = eval( subs(dfdx) );

            % Reset to initial state and input
            ps.set_state(x_init);
            ps.set_input(u_init);

        end

        function [ B_out ] = B1(ps,x,u)
            %B1
            %Description:
            %   Computes the linearized system matrix B assuming that the motion cone is the 'sticking one'
            %Usage:
            %   [ B ] = ps.B1(x,u)
            %

            % Constants

            x_init = ps.x();
            u_init = ps.u();

            % Algorithm

            ps.set_state(x);

            currMode = ps.identify_mode(u);

            syms s_x s_y s_theta p_y v_n v_t real;

            % Identify A
            symvar_x = [s_x; s_y; s_theta; p_y];
            symvar_u = [v_n; v_t];

            % (Mode is assumed to be 1 (sticking))
            % Compute Jacobian
            dxdt = ps.f1(symvar_x,symvar_u);
            dfdu = jacobian(dxdt,symvar_u);

            s_x = x(1); s_y = x(2); s_theta = x(3); p_y = x(4);
            v_n = u(1); v_t = u(2);

            B_out = eval( subs(dfdu) );

            % Reset to initial state and input
            ps.set_state(x_init);
            ps.set_input(u_init);

        end

        function [ B_out ] = B2(ps,x,u)
            %B2
            %Description:
            %   Computes the linearized system matrix B assuming that the motion cone is the 'sliding up' one
            %Usage:
            %   [ B ] = ps.B2(x,u)
            %

            % Constants

            x_init = ps.x();
            u_init = ps.u();

            % Algorithm

            ps.set_state(x);

            currMode = ps.identify_mode(u);

            syms s_x s_y s_theta p_y v_n v_t real;

            % Identify A
            symvar_x = [s_x; s_y; s_theta; p_y];
            symvar_u = [v_n; v_t];

            % (Mode is assumed to be 1 (sticking))
            % Compute Jacobian
            dxdt = ps.f2(symvar_x,symvar_u);
            dfdu = jacobian(dxdt,symvar_u);

            s_x = x(1); s_y = x(2); s_theta = x(3); p_y = x(4);
            v_n = u(1); v_t = u(2);

            B_out = eval( subs(dfdu) );

            % Reset to initial state and input
            ps.set_state(x_init);
            ps.set_input(u_init);

        end

        function [ B_out ] = B3(ps,x,u)
            %B3
            %Description:
            %   Computes the linearized system matrix B assuming that the motion cone is the 
            %   'sliding down' one
            %Usage:
            %   [ B ] = ps.B3(x,u)
            %

            % Constants

            x_init = ps.x();
            u_init = ps.u();

            % Algorithm

            ps.set_state(x);

            currMode = ps.identify_mode(u);

            syms s_x s_y s_theta p_y v_n v_t real;

            % Identify A
            symvar_x = [s_x; s_y; s_theta; p_y];
            symvar_u = [v_n; v_t];

            % (Mode is assumed to be 1 (sticking))
            % Compute Jacobian
            dxdt = ps.f3(symvar_x,symvar_u);
            dfdu = jacobian(dxdt,symvar_u);

            s_x = x(1); s_y = x(2); s_theta = x(3); p_y = x(4);
            v_n = u(1); v_t = u(2);

            B_out = eval( subs(dfdu) );

            % Reset to initial state and input
            ps.set_state(x_init);
            ps.set_input(u_init);

        end

        function [ C_t , C_b ] = LinearizedMotionConePerturbationVectorsAbout(ps, x, u)
            %LinearizedMotionConeVectors
            %Description:
            %
            %Usage:
            %   [ C_t , C_b ] = ps.LinearizedMotionConePerturbationVectorsAbout(x,u)

            % Constants
            g = 10;
            f_max = ps.st_cof * ps.s_mass*g;
            m_max = ps.st_cof * ps.s_mass*g * (ps.s_width/2); % The last term is meant to come from a sort of mass distribution/moment calculation. ???
            c = f_max / m_max;
            mu = ps.st_cof; %Which coefficient of friction is this supposed to be?

            % Algorithm

            %Find Symbolic form of gradient
            syms s_x s_y s_theta p_y v_n v_t real;

            symvar_x = [s_x; s_y; s_theta; p_y];
            symvar_u = [v_n; v_t];

            gamma_t = (mu*c.^2 - ps.p_x * p_y + mu*ps.p_x^2)/( c.^2 + p_y.^2-mu*ps.p_x*p_y );
            gamma_b = (-mu*c.^2 - ps.p_x * p_y - mu*ps.p_x^2)/( c.^2 + p_y.^2 + mu*ps.p_x*p_y );

            dg_tdx = gradient(gamma_t, symvar_x);
            dg_bdx = gradient(gamma_b, symvar_x);

            s_x = x(1); s_y = x(2); s_theta = x(3); p_y = x(4);
            v_n = u(1); v_t = u(2);

            C_t = eval( subs(dg_tdx) );
            C_b = eval( subs(dg_bdx) );

        end

        function [ E1 , D1 , g1 ] = Linearized_StickingInteractionMatrices_About( ps , x , u )
            %Linearized_StickingInteractionMatrices_About
            %Description:
            %   Computes the matrices which define the linearized condition for sticking of the pusher slider
            %   interaction.
            %       E1(t) bar_x + D1(t) bar_u <= g1(t)
            %   Note that the output of this function is E1, D1, d1 at a specific time when the state is x
            %   and the input is u.
            %
            %Usage:
            %   [ E1 , D1 , g1 ] = ps.Linearized_StickingInteractionMatrices_About( x , u )

            % Constants
            x_init = ps.get_state();
            u_init = ps.get_input();

            ps.set_state(x);
            ps.set_input(u);
            [ C_t , C_b ] = ps.LinearizedMotionConePerturbationVectorsAbout(x,u);
            [gamma_t_star, gamma_b_star] = ps.get_motion_cone_vectors();

            x_star = x;
            u_star = u;

            v_n_star = u_star(1); v_t_star = u_star(2);

            % Algorithm

            E1 = v_n_star * [ -C_t' ; C_b' ];
            D1 = [ -gamma_t_star, 1 ; gamma_b_star, -1 ];
            g1 = [ -v_t_star + gamma_t_star*v_n_star ; v_t_star - gamma_b_star * v_n_star ];

            % Restore State to Initial Values
            ps.set_state(x_init);
            ps.set_input(u_init);

        end

        function [ E2 , D2 , g2 ] = Linearized_SlidingUpInteractionMatrices_About( ps , x , u )
            %Linearized_SlidingUpInteractionMatrices_About
            %Description:
            %   Computes the matrices which define the linearized condition for sliding down mode
            %   of the pusher slider interaction.
            %       E2(t) bar_x + D2(t) bar_u <= g2(t)
            %   Note that the output of this function is E1, D1, d1 at a specific time when the state is x
            %   and the input is u.
            %
            %Usage:
            %   [ E2 , D2 , g2 ] = ps.Linearized_SlidingUpInteractionMatrices_About( x , u )

            % Constants
            x_init = ps.get_state();
            u_init = ps.get_input();

            ps.set_state(x);
            ps.set_input(u);
            [ C_t , C_b ] = ps.LinearizedMotionConePerturbationVectorsAbout(x,u);
            [gamma_t_star, gamma_b_star] = ps.get_motion_cone_vectors();

            x_star = x;
            u_star = u;

            v_n_star = u_star(1); v_t_star = u_star(2);

            eps0 = 10^(-7);

            % Algorithm

            E2 = v_n_star * C_t';
            D2 = [ gamma_t_star, -1 ];
            g2 = [ v_t_star - gamma_t_star*v_n_star - eps0 ];

            % Restore State to Initial Values
            ps.set_state(x_init);
            ps.set_input(u_init);

        end

        function [ E3 , D3 , g3 ] = Linearized_SlidingDownInteractionMatrices_About( ps , x , u )
            %Linearized_SlidingDownInteractionMatrices_About
            %Description:
            %   Computes the matrices which define the linearized condition for sliding down mode
            %   of the pusher slider interaction.
            %       E3(t) bar_x + D3(t) bar_u <= g3(t)
            %   Note that the output of this function is E1, D1, d1 at a specific time when the state is x
            %   and the input is u.
            %
            %Usage:
            %   [ E3 , D3 , g3 ] = ps.Linearized_SlidingDownInteractionMatrices_About( x , u )

            % Constants
            x_init = ps.get_state();
            u_init = ps.get_input();

            ps.set_state(x);
            ps.set_input(u);
            [ C_t , C_b ] = ps.LinearizedMotionConePerturbationVectorsAbout(x,u);
            [gamma_t_star, gamma_b_star] = ps.get_motion_cone_vectors();

            x_star = x;
            u_star = u;

            v_n_star = u_star(1); v_t_star = u_star(2);

            eps0 = 10^(-7);

            % Algorithm

            E3 = - v_n_star * C_b';
            D3 = [ -gamma_b_star, 1 ];
            g3 = [ -v_t_star + gamma_b_star*v_n_star - eps0 ];

            % Restore State to Initial Values
            ps.set_state(x_init);
            ps.set_input(u_init);

        end

    end
end

