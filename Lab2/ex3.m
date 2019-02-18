% LAB 2 EX 3 
rawData=importdata('XwindowsDocData.mat'); %call for import the file name
load('XwindowsDocData.mat', 'xtrain');
load('XwindowsDocData.mat','ytrain');
load('XwindowsDocData.mat', 'xtest');
load('XwindowsDocData.mat','ytest');
yTotal= size(ytrain);

[xtestTotal,xtestClTotal] =size(xtest); % Calculate the Number of columns and Rows
[ytestTotal,ytestClTotal]=size(ytest);  % Calculate the Number of columns and Rows

[xtrainTotal,xtrainClTotal] =size(xtrain); % Calculate the Number of columns and Rows
[ytrainTotal,ytrainClTotal]=size(ytrain);  % Calculate the Number of columns and Rows

% calculation of probabilty of MS windows 
msW = ytrain(:,1)==2;
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

Thetajc = Njc ./ [xWTotal(1) msWTotal(1)];%class-conditional probablity


% TEST DATA

% Calculate MAP estimate of the class the vector belong to

sum_cTest = zeros(xtestTotal,2); % log(p(x|y=c) + logp(c)
for i=1:xtestTotal % iterate through Xtest to calculate p(x|y=c)
    for cl=1:xtestClTotal
        
%         if xtest(i,cl)==1
%             sum_c(i,:) = sum_c(i,:) + log(Thetajc(cl,:)); % log(thetajc)= logp(y=c|x)
%         else 
%             sum_c(i,:) = sum_c(i,:) + log(1- Thetajc(cl,:));
%         end
        x= xtest(i,cl);
        sum_cTest(i,:)= sum_cTest(i,:) + ((log(Thetajc(cl,:))).^x).*(log(1- Thetajc(cl,:)).^(1-x));
    end
end

% Calculate the MAP estimate logP(x|y=c)+logP(c)
mapEstimate=zeros(xtestTotal,1);
for i=1:xtestTotal
    [~,c] = max([sum_cTest(i,1),sum_cTest(i,2)]);
    mapEstimate(i,1) = c;
end

%Compare the results with ytest and calculate the error percentage 
totalCorrectTest = sum(ytest == mapEstimate); % Calculate the total number of correct from a 
                                          %logical vector resulting from
                                          %the comparison of two vectors
testAccuracy = (totalCorrectTest / xtestTotal)*100; % calculate the accuracy of the computation

% TRAIN DATA
% Calculate MAP estimate of the class the vector belong to

sum_cTrain = zeros(xtrainTotal,2); % log(p(x|y=c) + logp(c)

for i=1:xtrainTotal % iterate through Xtest to calculate p(c|y=c)
    for cl=1:xtrainClTotal
        
%         if xtest(i,cl)==1
%             sum_c(i,:) = sum_c(i,:) + log(Thetajc(cl,:)); % log(thetajc)= logp(y=c|x)
%         else 
%             sum_c(i,:) = sum_c(i,:) + log(1- Thetajc(cl,:));
%         end
        x= xtrain(i,cl);
        sum_cTrain(i,:)= sum_cTrain(i,:) + ((log(Thetajc(cl,:))).^x).*(log(1- Thetajc(cl,:)).^(1-x));
    end
end

% Calculate the MAP estimate logP(x|y=c)+logP(c)
mapEstimate=zeros(xtrainTotal,1);
for i=1:xtrainTotal
    [~,c] = max([sum_cTrain(i,1),sum_cTrain(i,2)]);
    mapEstimate(i,1) = c;
end

%Compare the results with ytest and calculate the error percentage 
totalCorrectTrain = sum(ytrain == mapEstimate); % Calculate the total number of correct from a 
                                          %logical vector resulting from
                                          %the comparison of two vectors
trainAccuracy = (totalCorrectTrain / xtrainTotal)*100; % calculate the accuracy of the computation

% Compare the Accuracy of Test with Train

compAccuracy = abs(trainAccuracy-testAccuracy);

%% Optional part

%Save prior probabilities in a single vector for convenience
pi = [ppxW,ppmsW];

%First compute Thetaj
Thetaj = zeros(600,1);

for j=1:xtrainClTotal
    Thetaj(j) = sum(Thetajc(j,:).*pi);
end

Ij = zeros(xtrainClTotal,1);
for j=1:xtrainClTotal
    runningsum = 0;
    for c=1:2
        A = Thetajc(j,c)*pi(c) * log(Thetajc(j,c)/(Thetaj(j)+eps));
        B = (1-Thetajc(j,c))*pi(c) * log((1-Thetajc(j,c))/(1-Thetaj(j))+eps);
        runningsum = runningsum + A + B;    
    end
    Ij(j) = runningsum;
end

[Values,Index] = sort(Ij,'descend');

K  = 200;
accuracy_list = zeros(1,1);

for k=1 :5: K

mostImportantK = Index(1:k);
% Calculate MAP estimate of the class the vector belong to

sum_cK = zeros(xtestTotal,2); % log(p(x|y=c) + logp(c)
for i=1:xtestTotal % iterate through Xtest to calculate p(c|y=c)
    for cl=1:k
        
%         if xtest(i,cl)==1
%             sum_c(i,:) = sum_c(i,:) + log(Thetajc(cl,:)); % log(thetajc)= logp(y=c|x)
%         else 
%             sum_c(i,:) = sum_c(i,:) + log(1- Thetajc(cl,:));
%         end
        x= xtest(i,mostImportantK(cl));
        sum_cK(i,:)= sum_cK(i,:) + ((log(Thetajc(mostImportantK(cl),:))).^x).*(log(1- Thetajc(mostImportantK(cl),:)).^(1-x));
    end
end

% Calculate the MAP estimate logP(x|y=c)+logP(c)
mapEstimateK=zeros(xtestTotal,1);
for i=1:xtestTotal
    [~,c] = max([sum_cK(i,1),sum_cK(i,2)]);
    mapEstimateK(i,1) = c;
end

%Compare the results with ytest and calculate the error percentage 
totalCorrectK = sum(ytest == mapEstimateK); % Calculate the total number of correct from a 
                                          %logical vector resulting from
                                          %the comparison of two vectors                            
kAccuracy = (totalCorrectK / xtestTotal)*100; % calculate the accuracy of the computation
accuracy_list = [accuracy_list kAccuracy];
end
plot(1:5:K,accuracy_list(1,2:size(1:5:K,2)+1));
title("Accuracy vs K Important features");
xlabel('K Important Features'); ylabel('Accuracy (%)');