classdef simulationTest < matlab.unittest.TestCase
    
    methods(TestClassSetup)
        % Shared setup for the entire test class
    end
    
    methods(TestMethodSetup)
        % Setup for each test
    end
    
    methods(Test)
        % Test methods

        function basicTimeDerivativeTest1(testCase)
            %Description:
            %   In this test, we will make sure that the Pusher Slider and the
            %   PusherSlider with Unknown CoM have the same derivative/dynamics output
            %   when given equivalent states.
            %   In this case, offset is zero. The two states should be treated the same.

            % Constants
            ps   = PusherSlider();
            ps_u = PusherSliderUnknownCoM();

            x0 = ps.x();
            u0 = [0.1;0];

            % Force offset to be zero.
            ps_u.p_y_offset = 0;

            % Compute derivative for each
            testCase.assertTrue(all(ps.f(x0,u0) == ps_u.f(x0,u0)) )

        end
        
        function basicTimeDerivativeTest2(testCase)
            %Description:
            %   In this test, we will make sure that the Pusher Slider and the
            %   PusherSlider with Unknown CoM have the same derivative/dynamics output
            %   when given equivalent states.
            %   In this case, offset is nonzero.

            % Constants
            ps   = PusherSlider();
            ps_u = PusherSliderUnknownCoM();

            x0 = ps.x();
            u0 = [0.1;0];

            % Force offset to be zero.
            ps_u.p_y_offset = 0.1;

            % Compute derivative for each
            testCase.assertTrue(all(ps.f(x0 + [0;0;0;0.1],u0) == ps_u.f(x0,u0)) )

        end

        function symbolicDerivativeTest1(testCase)
            %Description:
            %   Computes the symbolic derivative of the nonlinear system's mode 1 (sticking)
            %   with respect to the derivative.

            % Constants
            ps_u = PusherSliderUnknownCoM();
            sym_offset = sym('s_o','real');
            sym_state  = sym('x',[4,1],'real');
            sym_input  = sym('u',[2,1],'real');

            ps_u.p_y_offset = sym_offset;

            % ps_u.f1(sym_state,sym_input)

            %% Compute gradient with respect to theta.

            diff( ps_u.f1(sym_state,sym_input) , sym_offset )

        end

    end
    
end