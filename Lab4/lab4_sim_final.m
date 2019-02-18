% Lab 4
clear all
close all

%% Simulation of object motion

N=100; % number of time instants

% System configuration
Delta=1.5;
A=[1 0 Delta 0; 0 1 0 Delta; 0 0 1 0; 0 0 0 1];

% Defining system noise
sigma_Qx= 1; % System noise - location
sigma_Qv= 0.5; % System noise - velocity
epsilon=zeros(4,N);
epsilon(1:2,:)=sigma_Qx*randn(2,N);
epsilon(3:4,:)=sigma_Qv*randn(2,N);

% Generating trajectory
z=zeros(4,N); % state vector (over time)
z(:,1)=[0 0 Delta Delta].'; % Initial state: coordinates at time 0 are (0,0)
for i=2:N
    z(:,i)=A*z(:,i-1)+epsilon(:,i);
end

C=[1 0 0 0 ; 0 1 0 0];
% Defining measurement noise
sigma_R=20; % Measurement noise

% Generate measurements of true trajectory
delta=sigma_R*randn(2,N);
y=zeros(2,N);
y(:,1)=[0 0].';
for i=2:N
    y(:,i)=C*z(:,i)+delta(:,i);
end

%% Kalman filter

% Define Q and R
Q = diag([sigma_Qx,sigma_Qx,sigma_Qv,sigma_Qv]); % System Noise Covariance Matrix
R = diag([sigma_R,sigma_R]); % measurment noise Covariance Matrix

% Set initial values
Delta_initial = 5;
mu = [0;0;Delta_initial;Delta_initial];
Sigma = diag(diag(ones(4))); %Identity matrix

% (Optionally) change A
%Delta_changed = 30;
Delta_changed = Delta; %Keep it the same
A=[1 0 Delta_changed 0; 0 1 0 Delta_changed; 0 0 1 0; 0 0 0 1];

% Initialize vector of predicted locations
y_hat = zeros(2,N);

% Run the Kalman filter
for i=1:N
    % Prediction phase
    mu = A*mu;
    Sigma = A*Sigma*A' + Q;
    
    % Updating phase
    y_hat(:,i) = C*mu;
    K = Sigma*C' * inv(C*Sigma*C' + R);
    r = y(:,i) - y_hat(:,i);
    mu = mu + K*r;
    Sigma = (diag(diag(ones(4))) - K*C)*Sigma;
end

% Figures

% This figure plots object motion trajectory and the Kalman estimation
% figure
% subplot(1,2,1)  
% plot(z(1,:),z(2,:))
% hold on
% plot(y(1,:),y(2,:))
% hold on
% plot(y_hat(1,:),y_hat(2,:))
% legend('True coordinates','Observed coordinates','Estimated coordinates');
% xlabel('X coordinate')
% ylabel('Y coordinate')
% title('Object x- and y-coordinate')

% This figure plots object x coordinate and the Kalman estimation
% subplot(1,2,2) 
figure
plot(1:N,z(1,:))
hold on
plot(1:N,y(1,:))
hold on
plot(1:N,y_hat(1,:))
legend('True coordinate','Observed coordinate','Estimated coordinate');
xlabel('Time instance')
ylabel('X coordinate')
title('Object x-coordinate vs. time')
hold off
