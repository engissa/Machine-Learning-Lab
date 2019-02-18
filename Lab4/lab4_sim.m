% Lab 4

clear all
close all

% Simulation of object motion

N=100; % number of time instants

Delta=1.5; % Velocity of the object 

A=[1 0 Delta 0; 0 1 0 Delta; 0 0 1 0; 0 0 0 1];

sigma_Qx= 2;
sigma_Qv= 0.5;

epsilon=zeros(4,N);
epsilon(1:2,:)=sigma_Qx*randn(2,N);
epsilon(3:4,:)=sigma_Qv*randn(2,N);
z=zeros(4,N); % state vector (over time)
z(:,1)=[0 0 Delta Delta].'; % Initial state: coordinates at time 0 are (0,0)
for i=2:N
    z(:,i)=A*z(:,i-1)+epsilon(:,i);
end


C=[1 0 0 0 ; 0 1 0 0];
sigma_R=20;
delta=sigma_R*randn(2,N);
y=zeros(2,N);
y(:,1)=[0 0].';
for i=2:N
    y(:,i)=C*z(:,i)+delta(:,i);
end

% This figure plots object motion trajectory
figure
plot(z(1,:),z(2,:))
