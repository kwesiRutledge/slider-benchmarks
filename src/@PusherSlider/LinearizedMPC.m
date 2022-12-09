function [ u_bar_star , x_bar_star , opt_data ] = LinearizedMPC( varargin )
	%Description:
	%	Computes the T-step Model Predictive Control output (u_traj) which is a T-length sequence
	%	of inputs for the PusherSlider system. The controller models the nonlinear system with a
	%	linear, time-varying system that is linearized about the points x_traj. defines a linear
	%
	%Usage:
	%	[ u_bar_star , x_bar_star , opt_data ] = ps.LinearizedMPC( x0 , T , x_traj , u_traj , dt )
	%	[ u_bar_star , x_bar_star , opt_data ] = ps.LinearizedMPC( x0 , T , x_traj , u_traj , dt , 'Q' , Q , 'R' , R , 'Q_T' , Q_T )
	%	[ u_bar_star , x_bar_star , opt_data ] = ps.LinearizedMPC( x0 , T , x_traj , u_traj , dt , 'verbosity' , 0 )

	%%%%%%%%%%%%%%%%%%%%%%
	%% Input Processing %%
	%%%%%%%%%%%%%%%%%%%%%%

	[ ps , x0 , T , x_traj , u_traj , dt , Q , R , Q_T , verbosity , warning_val ] = LinearizedMPC_InputProcessing( varargin{:} );

	%%%%%%%%%%%%%%%%%%
	%% Optimization %%
	%%%%%%%%%%%%%%%%%%

	%x0 = x_traj(:,1);
	u0 = u_traj(:,1);

	dim_u = 2;
	h = dt;

	T = 10; %Time Horizon for the MPC Problem

	%% Create Optimization Variables

	u_bar = {};
	for t = 0:T-1
		u_bar{t+1} = sdpvar(dim_u,1,'full');
	end

	active_cone_at = {};
	for t = 0:T-1
		active_cone_at{t+1} = binvar(3,1,'full');
	end

	dim_x = 4;
	x_at = {};
	for t = 0:T
		x_at{t+1} = sdpvar(dim_x,1,'full');
	end

	%%%%%%%%%%%%%%%%%%%%%%%%
	%% Create Constraints %%
	%%%%%%%%%%%%%%%%%%%%%%%%

	initial_state_constraint = [ x0 == x_at{1} ];

	[ A_j , B_j ] = ps.LinearizedSystemAbout(x0,u0);

	% Evaluate f_j0 at different values of j
	f_j0{1} = (x0 - x_traj(:,1)) + h*( ps.B1(x0,u0) * u_traj(:,1) - ps.f(x_traj(:,1),u_traj(:,1)) );
	f_j0{2} = (x0 - x_traj(:,1)) + h*( ps.B2(x0,u0) * u_traj(:,1) - ps.f(x_traj(:,1),u_traj(:,1)) );
	f_j0{3} = (x0 - x_traj(:,1)) + h*( ps.B3(x0,u0) * u_traj(:,1) - ps.f(x_traj(:,1),u_traj(:,1)) );

	[ E1 , D1 , g1 ] = ps.Linearized_StickingInteractionMatrices_About( x_traj(:,1) , u_traj(:,1) );
	[ E2 , D2 , g2 ] = ps.Linearized_SlidingUpInteractionMatrices_About( x_traj(:,1) , u_traj(:,1) );
	[ E3 , D3 , g3 ] = ps.Linearized_SlidingDownInteractionMatrices_About( x_traj(:,1) , u_traj(:,1) );

	initial_motion_cone1_constraint = [ x_at{2} == f_j0{1} + h*ps.B1(x0,u0) * u_bar{1} ] + ...
		[ D1 * u_bar{1} <= g1 ];
	initial_motion_cone2_constraint = [ x_at{2} == f_j0{2} + h*ps.B2(x0,u0) * u_bar{1} ] + ...
		[ D2 * u_bar{1} <= g2 ];
	initial_motion_cone3_constraint = [ x_at{2} == f_j0{3} + h*ps.B3(x0,u0) * u_bar{1} ] + ...
		[ D3 * u_bar{1} <= g3 ];

	initial_motion_cone_selection_constraint = ...
		[ implies(active_cone_at{1}(1) == 1, initial_motion_cone1_constraint) ] + ...
		[ implies(active_cone_at{1}(2) == 1, initial_motion_cone2_constraint ) ] + ...
		[ implies(active_cone_at{1}(3) == 1, initial_motion_cone3_constraint ) ] + ...
		[ sum(active_cone_at{1}) == 1 ];

	rest_of_motion_constraints = [];
	for t = 1:T-1 %Should start from t=1 (i.e. we already did the constraints from t=0 to t=1. We must start from constraints t=1 on...)

		% Get Constant Matrices at this instant in the ltrajectory
		[ E1 , D1 , g1 ] = ps.Linearized_StickingInteractionMatrices_About( x_traj(:,t+1) , u_traj(:,t+1) );
		[ E2 , D2 , g2 ] = ps.Linearized_SlidingUpInteractionMatrices_About( x_traj(:,t+1) , u_traj(:,t+1) );
		[ E3 , D3 , g3 ] = ps.Linearized_SlidingDownInteractionMatrices_About( x_traj(:,t+1) , u_traj(:,t+1) );

		% Create the individual constraints which hold in each motion cone
		% E2 * x_at{t+1} 

		motion_cone1_constraint_at_t = ...
			[ x_at{t+2} == (eye(dim_x) + h*ps.A1(x_traj(:,t+1),u_traj(:,t+1)) )*x_at{t+1} + h*ps.B1(x_traj(:,t+1),u_traj(:,t+1)) * u_bar{t+1} ] + ...
			[ E1 * x_at{t+1} + D1 * u_bar{t+1} <= g1 ];
		motion_cone2_constraint_at_t = ...
			[ x_at{t+2} == (eye(dim_x) + h*ps.A2(x_traj(:,t+1),u_traj(:,t+1)) )*x_at{t+1} + h*ps.B2(x_traj(:,t+1),u_traj(:,t+1)) * u_bar{t+1} ] + ...
			[ E2 * x_at{t+1} + D2 * u_bar{t+1} <= g2 ];
		motion_cone3_constraint_at_t = ...
			[ x_at{t+2} == (eye(dim_x) + h*ps.A3(x_traj(:,t+1),u_traj(:,t+1)) )*x_at{t+1} + h*ps.B3(x_traj(:,t+1),u_traj(:,t+1)) * u_bar{t+1} ] + ...
			[ E3 * x_at{t+1} + D3 * u_bar{t+1} <= g3 ];

		rest_of_motion_constraints = rest_of_motion_constraints + ...
			[ implies( active_cone_at{t+1}(1) == 1 , motion_cone1_constraint_at_t ) ] + ...
			[ implies( active_cone_at{t+1}(2) == 1 , motion_cone2_constraint_at_t ) ] + ...
			[ implies( active_cone_at{t+1}(3) == 1 , motion_cone3_constraint_at_t ) ] + ...
			[ sum(active_cone_at{t+1}) == 1 ];
	end

	% Create constraints on input u_bar
	push_only_constraint = [];
	cannot_lose_contact_constraint = [];
	for t = 0:T-1
		push_only_constraint = push_only_constraint + [ u_bar{t+1}(1) + u_traj(1,t+1) >= 0 ]; %Cannot exert negative push on slider (i.e. cannot pull it)
		cannot_lose_contact_constraint = cannot_lose_contact_constraint + [ -ps.s_width <= x_at{t+2}(4) <= ps.s_width ];
	end

	%%%%%%%%%%%%%%%%%%%%%%
	%% Create Objective %%
	%%%%%%%%%%%%%%%%%%%%%%

	objective = x_at{T+1}'*Q_T*x_at{T+1};
	for t = 0:T-1
		objective = objective + x_at{t+1}'*Q*x_at{t+1} + u_bar{t+1}'*R*u_bar{t+1};
	end

	%%%%%%%%%%%%%%
	%% Optimize %%
	%%%%%%%%%%%%%%

	ops0 = sdpsettings('verbose',verbosity,'warning',warning_val);

	opt_data = optimize( ...
		initial_state_constraint+initial_motion_cone_selection_constraint+rest_of_motion_constraints+...
		cannot_lose_contact_constraint+push_only_constraint, ...
		objective, ops0 );

	u_bar_star = [];
	for u_ind = 1:length(u_bar)
		u_bar_star = [ u_bar_star , value(u_bar{u_ind}) ];
	end

	x_bar_star = [];
	for x_ind = 1:length(x_at)
		x_bar_star = [ x_bar_star , value(x_at{x_ind}) ];
	end

end

function [ ps , x0 , T , x_traj , u_traj , dt , Q , R , Q_T , verbosity , warning_val ] = LinearizedMPC_InputProcessing( varargin )
	%Description:
	%	Checks the inputs which were given to the LinearizedMPC function.

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% Incorporate the Mandatory Arguments %%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	if nargin < 6
		error(['Expected at least 6 arguments. Received ' num2str(nargin) '.' ])
	end

	ps = varargin{1};
	x0 = varargin{2};
	T  = varargin{3};
	x_traj = varargin{4};
	u_traj = varargin{5};
	dt = varargin{6};

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% Checking For Extra Inputs %%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	if nargin > 6
		argidx = 7;
		while argidx <= nargin
			switch varargin{argidx}
				case 'Q'
					Q = varargin{argidx+1};
					argidx = argidx + 2;
				case 'R'
					R = varargin{argidx+1};
					argidx = argidx + 2;
				case 'Q_T'
					Q_T = varargin{argidx+1};
					argidx = argidx + 2;
				case 'verbosity'
					verbosity = varargin{argidx+1};
					argidx = argidx + 2;
				case 'warning_val'
					warning_val = varargin{argidx+1};
					argidx = argidx + 2;
				otherwise
					error(['Unexpected extra input to LinearizedMPC: ' varargin{argidx} ])
			end
		end
	end

	%%%%%%%%%%%%%%%%%%%%%%
	%% Setting Defaults %%
	%%%%%%%%%%%%%%%%%%%%%%

	if ~exist('Q')
		Q = eye(4);
	end

	if ~exist('R')
		R = eye(2);
	end

	if ~exist('Q_T')
		Q_T = eye(4);
	end

	if ~exist('verbosity')
		verbosity = 1;
	end

	if ~exist('warning_val')
		warning_val = 0; % By default ignore all warnings coming from YALMIP. (It usually complains about big M values.)
	end

	%%%%%%%%%%%%%%%%%%%%%
	%% Checking Inputs %%
	%%%%%%%%%%%%%%%%%%%%%

	%% Check x0

	if length(x0) ~= 4
		error(['Expected the input x0 to be four dimensional. Instead it has dimension ' num2str(length(x0)) '.' ])
	end

	%% Check T

	if T < 0
		error(['Expected time horizon to be a positive number. Received ' num2str(T) '.' ])
	end

	%% Check x_traj
	if size(x_traj,1) ~= 4
		error(['Expected x_traj to be a 4 x ' num2str(T) ' matrix. Received a ' num2str(size(x_traj,1)) ' x ' num2str(size(x_traj,2)) ' matrix.'  ])
	end

	if size(x_traj,2) ~= T
		error(['Expected x_traj to contain ' num2str(T) ' states. Received a matrix with ' num2str(size(x_traj,2)) ' states.'  ])
	end		

end