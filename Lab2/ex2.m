% LAB 2 EX 1 
rawData=importdata('XwindowsDocData.mat'); %call for import the file name
load('XwindowsDocData.mat', 'xtrain');
load('XwindowsDocData.mat','ytrain');
yTotal= size(ytrain);

% calculation of probabilty of MS windows
msW = ytrain(:,1)==2; % number
msWdataSet=ytrain(msW,:);
msWTotal=size(msWdataSet);
ppmsW= msWTotal/yTotal; % prior probabilty of MS windows 

% Calculation of probablity of windows X
xW= ytrain(:,1)==1;
xWdataSet=ytrain(xW,:);
xWTotal = size(xWdataSet);
ppxW= xWTotal/yTotal;% prior probability of windows X

% Fitting Naive Bayes classifier to binary features
Njc = ones(600,2);

for i=1:yTotal
    for j=1:600
        c = ytrain(i);
        if xtrain(i,j)==1
            Njc(j,c) = Njc(j,c)+1;
        end
    end
end

Thetajc = Njc ./ [xWTotal(1) msWTotal(1)];

%Append column with feature numbers
ThetajcA = [(1:600)' Thetajc];
% bar(ThetajcA(:,1), ThetajcA(:,2));
figure()
bar(ThetajcA(:,1),ThetajcA(:,3));
title('class-conditional densities of Class 1');
xlabel('Features'); ylabel('Probabilty');
figure()
bar(ThetajcA(:,1),ThetajcA(:,2));
title('class-conditional densities of Class 2');
xlabel('Features'); ylabel('Probabilty');
figure()
bar(ThetajcA(:,1),abs(ThetajcA(:,2)-ThetajcA(:,3)))
title('Uninformative Features');
xlabel('Features'); ylabel('Information');
%plot(ThetajcA(:,1), ThetajcA(:,2));