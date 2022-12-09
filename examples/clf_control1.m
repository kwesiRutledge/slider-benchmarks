%sos_control1.m
%Description:
%	Use a control law (verified by SOS) to steer the system towards a goal location (origin?).

%% Include Libraries
%  Make sure to include all files in the src directory and its subdirectories.
%	addpath(genpath('../src/'))

clear all;
close all;
clc;

%% Constants

g = 10;

%% Identify CLF

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

f_max = st_cof * s_mass*g;
m_max = st_cof * s_mass*g * (s_width/2); % The last term is meant to come from a sort of mass distribution/moment calculation. ???
c = f_max / m_max;
mu = st_cof; %Which coefficient of friction is this supposed to be?

% Define SOS Program
% ==================

% Create state variables
syms s_x s_y s_theta p_y s3 c3 real

Program1 = sosprogram([s_x, s_y, s_theta, p_y, s3, c3])
monoms_s_x = monomials([s_x],[1,2]);
monoms_s_y = monomials([s_y],[1,2]);
monoms_s_theta = monomials([s_theta],[1,2]);
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
tempExpr = - ( dV_dx' * [ C' * Q * P1 ; b1; c1; d1 ] * [ -1 * (s_x) ; -1 * (s_y)  ] ) - l2
Program1 = sosineq( Program1 , ...
    tempExpr ...
); %F3

% Optimize
Program1 = sossolve( Program1 );

assert( (Program1.solinfo.info.pinf == 0) || (Program1.solinfo.info.dinf == 0) );

%Plot SOL_V
Vsol = sosgetsol(Program1,V) %Getting solution for V
% Can't plot this for functions over 6 variables.

% Simulate System
% ===============

ps1 = PusherSlider();

x0s = [ [ -0.1, -0.1, pi/3, 0.02 ]' , [ -0.1, -0.1, pi/6, 0.02 ]' , [ -0.2, -0.1, pi/6, 0.02 ]' , [ -0.2, -0.1, 0, 0.02 ]' ];

dVsol_dx(1,1) = diff(Vsol,s_x);
dVsol_dx(2,1) = diff(Vsol,s_y);
dVsol_dx(3,1) = diff(Vsol,s_theta);
dVsol_dx(4,1) = diff(Vsol,p_y);
dVsol_dx(5,1) = diff(Vsol,s3);
dVsol_dx(6,1) = diff(Vsol,c3);

% Initializing Loop
T = 2;
dt = 0.1;
x_h = {}; V_h = {}; t_h = {};
for trial_index = [1:size(x0s,2)]
	% Get x0
	x0 = x0s(:,trial_index);

	x_t = x0;
	x = x_t;

	z0 = x0;
	z0(5,1) = sin(z0(3,1));
	z0(6,1) = cos(z0(3,1));
	z_t = z0;
	z = z_t;

	V = subs(Vsol,vectorToSubsStruct(z0));
	t_ti = [];
	for t = [0:dt:T-dt]
		% Get Current State And Evolve It with The Controller Output
		[u_t, opt_out_t] = CLF_input(x_t,Vsol,dVsol_dx,ps1)

		t_ti = [ t_ti , opt_out_t.yalmiptime ]

		ps1.f1( x_t , u_t )
		x_tp1 = x_t + dt * ps1.f1( x_t , u_t )
		x = [x, x_tp1];

		z_tp1 = x_tp1;
		z_tp1(5,1) = sin(z_tp1(3,1));
		z_tp1(6,1) = cos(z_tp1(3,1));
		z = [z, z_tp1];

		V_tp1 = subs(Vsol,vectorToSubsStruct(z_tp1));
		V = [V, V_tp1];

		% Prepare for next loop
		x_t = x_tp1;

	end

	%Add trajectory to the list.
	x_h{trial_index} = x;
	V_h{trial_index} = V;
	t_h{trial_index} = t_ti;
end

% Plot Results
figure;
subplot(2,1,1)
for trial_index = [1:length(V_h)]
	plot([0:dt:T],V_h{trial_index})
end
xlabel('Time')
ylabel('V')
title('Lyapunov Function Over Time')

% Write data to file.
fileID = fopen(['../data/clf_control1/clf_control1_results_' datestr(now(),'yyyymmdd-HHMMSS') '.txt'],'a');
fprintf(fileID, '%s \n',sum(Program1.solinfo.info.timing) )
for trial_index = [1:size(x0s,2)]
	
	timing_format_string = [];

	for t = [0:dt:T-dt]

		timing_format_string = [timing_format_string '%s '];

	end
	timing_format_string = [timing_format_string '\n'];

	%Write to file
	fprintf(fileID,timing_format_string, t_h{trial_index});

end
fclose(fileID);


%% Function Definitions

function [ subsStruct ] = vectorToSubsStruct( zIn )
	%vectorToSubStruct
	%Description:
	%	Converts a six dimensional vector (representing a state in the extended space z) to a
	%	struct that subs() can use to substitute values in a symbolic expression.

	% Constants

	% Algorithm
	subsStruct = struct(...
		's_x',zIn(1), ...
		's_y',zIn(2), ...
		's_theta',zIn(3), ...
		'p_y',zIn(4), ...
		's3',zIn(5), ...
		'c3',zIn(6) ...
		);

end

function [ u , opt_out ] = CLF_input( x , V , dVdx , psIn )
	%CLF_input
	%Description:
	%	Computes an allowable CLF input for the Pusher-Slider System given by psIn.

	% Constants
	z = x;
	z(5,1) = sin(z(3));
	z(6,1) = cos(z(3));

	% V_z = subs(V,vectorToSubsStruct(z))
	dVdx_z = subs(dVdx,vectorToSubsStruct(z));

	opt_settings = sdpsettings('verbose',0);

	g = 10;
    f_max = psIn.st_cof * psIn.s_mass*g;
    m_max = psIn.st_cof * psIn.s_mass*g * (psIn.s_width/2); % The last term is meant to come from a sort of mass distribution/moment calculation. ???
    c = f_max / m_max;

    eps0 = 1e-6;
    eta_u = 0.1;

	% Optimization
	u_var = sdpvar(2,1); % constants

	psIn.set_state(x);

	b1 = [( - psIn.p_y ) , psIn.p_x * (c.^2+psIn.p_x.^2+psIn.p_y.^2) ];
	c1 = zeros(1,2);
	d1 = zeros(2);

	% [ psIn.C()' * psIn.Q() * eye(2) ; b1 ; c1 ; d1 ] * u_var
	% dVdx_z' * [ psIn.C()' * psIn.Q() * eye(2) ; b1 ; c1 ; d1 ]
	% dVdx_z' * [ psIn.C()' * psIn.Q() * eye(2) ; b1 ; c1 ; d1 ] * u_var

	derivative_decrease_constraint = [ double(dVdx_z' * [ psIn.C()' * psIn.Q() * eye(2) ; b1 ; c1 ; d1 ]) * u_var <= -eps0 ];
	input_constraint = [ -eta_u <= u_var ] + [ u_var <= eta_u ];

	opt_out = optimize( derivative_decrease_constraint + input_constraint , norm(u_var) , opt_settings);

	% Create output
	u = value(u_var);

end