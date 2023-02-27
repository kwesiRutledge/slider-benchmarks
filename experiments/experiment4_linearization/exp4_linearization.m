%exp4_linearization
%Description
%   In this script, we attempt to "linearize" the Pusher Slider system
%   into something that we can reason about using our CAPA2 representation.
%   That is, we would like to represent it with the form:
%       \dot{x} = f(x) + F(x) \theta + (g(x) + \sum_i \theta_i G_i) u
%

%% Preamble
clear all; close all; clc;

%% Include Src
addpath(genpath('../../src'))

%% Create a Pusher Slider Instance With Variable CoM
syms theta

ps1 = PusherSliderUnknownCoM(theta);

x0 = ps1.x();
u0 = [0.1;0.03];

dim_theta = length(theta);
dim_x = length(x0); % Should be 4

u1 = sym('u1_',[2,1],'real');
x1 = sym('x1_',[dim_x,1],'real');

%% Differentiate with respect to symbolic functions
f1_0 = ps1.f1(x0,u0);
f1_1 = ps1.f1(x1,u1);

f2_1 = ps1.f2(x1,u1);
f3_1 = ps1.f3(x1,u1);


% Differentiate
df1_dx = sym(zeros(dim_x));
for dim_index = 1:dim_x
    df1_dx(dim_index,:) = gradient(f1_1(dim_index),x1);
end

df1_dtheta = sym(zeros(dim_x,dim_theta));
for dim_index = 1:dim_x
    df1_dtheta(dim_index,:) = gradient(f1_1(dim_index),theta);
end

df2_dtheta = sym(zeros(dim_x,dim_theta));
for dim_index = 1:dim_x
    df2_dtheta(dim_index,:) = gradient(f2_1(dim_index),theta);
end

df3_dtheta = sym(zeros(dim_x,dim_theta));
for dim_index = 1:dim_x
    df3_dtheta(dim_index,:) = gradient(f3_1(dim_index),theta);
end

%% Sample Some values of each gradient

dfdx_0 = double(subs(df1_dx, ...
    struct('x1_1',x0(1),'x1_2',x0(2),'x1_3',x0(3),'x1_4',x0(4), ...
        'u1_1',u0(1), 'u1_2', u0(2), ...
        'theta', 0) ...
));

df1_dtheta_0 = double(subs(df1_dtheta, ...
    struct('x1_1',x0(1),'x1_2',x0(2),'x1_3',x0(3),'x1_4',x0(4), ...
        'u1_1',u0(1), 'u1_2', u0(2), ...
        'theta', 0) ...
));

df2_dtheta_0 = double(subs(df2_dtheta, ...
    struct('x1_1',x0(1),'x1_2',x0(2),'x1_3',x0(3),'x1_4',x0(4), ...
        'u1_1',u0(1), 'u1_2', u0(2), ...
        'theta', 0) ...
));

df3_dtheta_0 = double(subs(df3_dtheta, ...
    struct('x1_1',x0(1),'x1_2',x0(2),'x1_3',x0(3),'x1_4',x0(4), ...
        'u1_1',u0(1), 'u1_2', u0(2), ...
        'theta', 0) ...
));