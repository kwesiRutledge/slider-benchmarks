classdef clfTest < matlab.unittest.TestCase
    
    methods(TestClassSetup)
        % Shared setup for the entire test class
    end
    
    methods(TestMethodSetup)
        % Setup for each test
    end
    
    methods(Test)
        % Test methods
        
        function basicSOSVariableTest(testCase)
            % Create a dummy program from the tutorial for SOSTools
            syms x y a b c d;
        
            Program1 = sosprogram([x;y]);
            [Program1,p1] = sossosvar(Program1,[x;y]);
            
            %p = 2*x^2 + 3 * x * y + 4 * y^4;
            p = 2*x^2 + 4 * y^4;

            %Add Constraints
            Program1 = sosineq( Program1 , p );

            Program1 = sossolve(Program1)
            Program1.solinfo.info.pinf

            testCase.verifyTrue( ...
                (Program1.solinfo.info.pinf == 0) && (Program1.solinfo.info.dinf == 0) ...
            );
        end

        %sosToolsCLFTest1
        %Description:
        %   This test attempts to verify if a standard CLF with a given feedback law
        %   is guaranteed to stabilize my simple system. (This should be the case.)
        function sosToolsCLFTest1(testCase)
            % Constants
            Theta1 = Polyhedron('lb',0.5,'ub',0.85);
            system1 = SimpleSystem1(Theta1)

            % Create a dummy program from the tutorial for YALMIP
            syms x a b c d;
        
            Program1 = sosprogram([x]);
            monom1 = monomials([x],[1,2,3,4]);
            [Program1,V] = sospolyvar(Program1,monom1);
            
            l1 = 0.01*x^2;
            l2 = 0.01*x^2;
            
            s1 = x^2;
            [Program1,p2] = sospolyvar(Program1,monom1);
            
            dV_dx = diff(V,x);
            
            % Create constraints
            Program1 = sosineq( Program1 , V - l1 ); %F2
            tempExpr = ...
                - ( dV_dx * ( system1.f(x) + system1.F(x)* system1.theta ) + ...
                (dV_dx*( system1.g(x) + system1.G(x) * system1.theta ) * ( -(1/(system1.theta)) - 1 )*x ) + l2)
            Program1 = sosineq( Program1 , ...
                tempExpr ...
            ); %F3

            % Optimize
            Program1 = sossolve( Program1 );

            testCase.verifyTrue( ...
                (Program1.solinfo.info.pinf == 0) || (Program1.solinfo.info.dinf ~= 0) ...
            );
        end

        function pusherSliderCLFTest1(testCase)
            %pusherSliderCLFTest1
            %Description:
            %   First attempt at using the standard CLF code to get a CLF for the standard pusher slider system.

            % Constants
            % =========

            %Define Default Physical Parameters
            s_width = 0.09;
            s_length = 0.09;
            s_mass = 1.05; %kg
            ps_cof = 0.3;
            st_cof = 0.35;

            p_radius = 0.01;

            p_x = s_width/2;

            g = 10;
            f_max = st_cof * s_mass*g;
            m_max = st_cof * s_mass*g * (s_width/2); % The last term is meant to come from a sort of mass distribution/moment calculation. ???
            c = f_max / m_max;
            mu = st_cof; %Which coefficient of friction is this supposed to be?

            % Define SOS Program
            % ==================

            % Create state variables
            syms s_x s_y s_theta p_y s3 c3 real
        
            Program1 = sosprogram([s_x, s_y, s_theta, p_y, s3, c3])
            monoms_s_x = monomials([s_x],[1,2])
            monoms_s_y = monomials([s_y],[1,2])
            monoms_s_theta = monomials([s_theta],[1,2])
            monoms_p_y = monomials([p_y],[1,2,3,4]);
            monoms_s3  = monomials([s3],[1,2,3,4]);
            monoms_c3  = monomials([c3],[1,2,3,4]);

            [Program1,V] = sospolyvar(Program1,...
                [monoms_s_x;monoms_s_y;monoms_s_theta;monoms_p_y;monoms_c3;monoms_s3]);
            
            l1 = 0.01*(s_x^2+s_y.^2+s_theta.^2+p_y.^2+s3.^2+c3.^2);
            l2 = 0.001*(s_x^2+s_y.^2+s_theta.^2+p_y.^2+s3.^2+c3.^2);
            
            s1 = (s_x^2+s_y.^2+s_theta.^2+p_y.^2+s3.^2+c3.^2);
            [Program1,p2] = sospolyvar(Program1,[monoms_s_x;monoms_s_y;monoms_s_theta;monoms_p_y;monoms_c3;monoms_s3]);

            % Create gradient vector dV_dx            
            dV_dx(1,1) = diff(V,s_x);
            dV_dx(2,1) = diff(V,s_y);
            dV_dx(3,1) = diff(V,s_theta);
            dV_dx(4,1) = diff(V,p_y);
            dV_dx(5,1) = diff(V,s3);
            dV_dx(6,1) = diff(V,c3);

            % Create dynamics matrices
            C = [c3,s3;-s3,c3];
            % Q = (1/ (c.^2+p_x.^2+p_y.^2) ) * [ c.^2 + p_x.^2, p_x * p_y; p_x * p_y, c.^2 + p_y.^2 ];
            Q = [ c.^2 + p_x.^2, p_x * p_y; p_x * p_y, c.^2 + p_y.^2 ];
            P1 = eye(2);

            %b1 = [( - p_y )/(c.^2+p_x.^2+p_y.^2) , p_x ];
            b1 = [( - p_y ) , p_x * (c.^2+p_x.^2+p_y.^2) ];
            c1 = zeros(1,2);
            d1 = zeros(2);
            
            % Create constraints
            Program1 = sosineq( Program1 , V - l1 ); %F2
            tempExpr = - ( dV_dx' * [ C' * Q * P1 ; b1; c1; d1 ] * [ -1 * (s_x) ; -1 * (s_y)  ] + l2)
            Program1 = sosineq( Program1 , ...
                tempExpr ...
            ); %F3

            % Optimize
            Program1 = sossolve( Program1 );

            testCase.verifyTrue( ...
                (Program1.solinfo.info.pinf == 0) || (Program1.solinfo.info.dinf == 0) ...
            );

            %Plot SOL_V
            SOL_V = sosgetsol(Program1,V) %Getting solution for V
            % figure;
            % ezplot(SOL_V)
        end
        
    end
    
end