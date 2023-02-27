%% point_estimate1.m
%   Description:
%       Uses data collected at a single state to estimate a flow map using
%       optimization.

clear all; close all; clc;

%% Load dataset
data_file = load('../../data/flowmap1/dataset_x_December132022-0513PM.mat')
dataset = data_file.dataset;

% ======================
% Construct Optimization

dim_x = size(dataset{1}.x,1);
dim_u = size(dataset{1}.u,1);

% Optimization Variables

A_over = sdpvar(dim_x,dim_x,'full');
B_over = sdpvar(dim_x,dim_u);
K_over = sdpvar(dim_x,1);

A_under = sdpvar(dim_x,dim_x,'full');
B_under = sdpvar(dim_x,dim_u);
K_under = sdpvar(dim_x,1);

% Create Constraints
constraints = [];
obj = 0;
for data_i = [1:length(dataset)]
    % Extract Data
    x_i = double(dataset{data_i}.x);
    u_i = double(dataset{data_i}.u);
    phi_i = double(dataset{data_i}.('phi(x,u,tau)'));

    % Add Constraints
    % - For Over-Approximation
    temp_constr = [ A_over * x_i + B_over * u_i + K_over >= phi_i ];
    constraints = constraints + temp_constr;
    % - For Under-Approximation
    constraints = constraints + [ A_under * x_i + B_under * u_i + K_under <= phi_i ];

    % Add TO Objective
    obj = obj + norm( A_over * x_i + B_over * u_i + K_over - phi_i ) + ...
        norm( A_under * x_i + B_under * u_i + K_under - phi_i );

end

% Optimize!
optim_out = optimize(temp_constr,obj)
disp('The model we''ve found is:')

A_u = value(A_under)
B_u = value(B_under)
K_u = value(K_under)
A_o = value(A_over)
B_o = value(B_over)
K_o = value(K_over)

save('point_estimate1_model.mat','A_o','B_o','K_o','A_u','B_u',"K_u")
