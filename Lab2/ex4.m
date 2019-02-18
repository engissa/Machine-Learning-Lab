%% LAB-2 Ex-1
rawData=importdata('heightWeight.mat'); %call for import the file name

males = rawData(:,1)==1;
maleDataSet=rawData(males,:);
malesTotal=size(maleDataSet,1);

malesTest = maleDataSet(1:25,:);
malesTrain = maleDataSet(26:malesTotal,:);
malesTrainTotal=size(malesTrain,1);
malesTestTotal= size(malesTest,1);

females = rawData(:,1)==2;
femaleDataSet=rawData(females,:);
femalesTotal=size(femaleDataSet,1);

femalesTest = femaleDataSet(1:35,:);
femalesTrain=femaleDataSet(36:femalesTotal,:);
femalesTrainTotal=size(femalesTrain,1);
femalesTestTotal= size(femalesTest,1);

figure();
subplot(1,2,1);
scatter(malesTest(:,2),malesTest(:,3),'bx');
grid on;
hold on;
scatter(femalesTest(:,2),femalesTest(:,3),'ro');
hold off;
title('Test DataSet');
xlabel('height');
ylabel('weight');
legend('Males', 'Females');

subplot(1,2,2);
scatter(malesTrain(:,2),malesTrain(:,3),'bx');
grid on;
hold on;
scatter(femalesTrain(:,2),femalesTrain(:,3),'ro');
hold off;
title('Train DataSet');
xlabel('height');
ylabel('weight');
legend('Males', 'Females');

totalTrain=femalesTrainTotal+malesTrainTotal;

priorProb=[(malesTrainTotal/totalTrain),(femalesTrainTotal/totalTrain)];

% MLE Classifier 
% 
% Calculate the mean of Male Height
meanHeightMale= sum(malesTrain(:,2))/malesTrainTotal;
% Calculate the mean of Male Weight 
meanWeightMale= sum(malesTrain(:,3))/malesTrainTotal;
% Vector containing the means of heigh and wieght of males
meanMale = [meanHeightMale meanWeightMale];

% Calculate the mean of Female Height 
meanHeightFemale= sum(femalesTrain(:,2))/femalesTrainTotal;
% Calculate the mean of Male Weight 
meanWeightFemale= sum(femalesTrain(:,3))/femalesTrainTotal;
% Vector containing the means of heigh and wieght of males
meanFemale = [meanHeightFemale meanWeightFemale];

runningMatrix = zeros(2,2); % Define a 2x2 empty matrix 
for i=1:malesTrainTotal
    runningMatrix = runningMatrix + malesTrain(i,2:3)'*malesTrain(i,2:3);
end
% Calculation of Sigma MLE for Males (covariance matrix)
sigma_males=(1/malesTrainTotal)*runningMatrix - meanMale'*meanMale;
% Calculation of MLE for females 
runningMatrix = zeros(2,2);% Define a 2x2 empty matrix 
for i=1:femalesTrainTotal
    runningMatrix = runningMatrix + femalesTrain(i,2:3)'*femalesTrain(i,2:3);
end
% Calculation of Sigma MLE for Females ( covarience matrix)
sigma_females=(1/femalesTrainTotal)*runningMatrix - meanFemale'*meanFemale;

% testing postDist function
cmalesMatrix = zeros(malesTestTotal,1);% Define

for i=1:malesTestTotal
cmalesMatrix(i)=postDist(malesTest(i,2:3),meanFemale,meanMale,sigma_males,sigma_females,priorProb);
end 

cfemalesMatrix = zeros(femalesTestTotal,1);% Define

for i=1:femalesTestTotal
cfemalesMatrix(i)=postDist(femalesTest(i,2:3),meanFemale,meanMale,sigma_males,sigma_females,priorProb);
end

figure();
subplot(2,2,1);
scatter(malesTest(cmalesMatrix(:,1)==1,2),malesTest(cmalesMatrix(:,1)==1,3),'bx');
grid on;
hold on;
scatter(femalesTest(cfemalesMatrix(:,1)==2,2),femalesTest(cfemalesMatrix(:,1)==2,3),'ro');
hold off;
title('Quadratic Discriminant Analysis');
xlabel('height');
ylabel('weight');
legend('Males', 'Females');

%Calculate Accuracy
accuracy_males = sum(cmalesMatrix(:)==1)/malesTestTotal;
accuracy_females = sum(cfemalesMatrix(:)==2)/femalesTestTotal;

%% Classifier 2

%Step 1: fit the training data to the parameters. The parameters for the
%mean are the same as the previous classifier:
mean_males2 = meanMale;
mean_females2 = meanFemale;

%Remove the off-diagonal entries in the covariance matrices for the new
%covariance matrices:
sigma_males2 = diag(diag(sigma_males));
sigma_females2 = diag(diag(sigma_females));

%Step 2: classify the test samples by calling the function with the new
%parameters
cmalesMatrix2 = zeros(malesTestTotal,1);
for i=1:malesTestTotal
cmalesMatrix2(i)=postDist(malesTest(i,2:3),mean_females2,mean_males2,sigma_males2,sigma_females2,priorProb);
end 

cfemalesMatrix2 = zeros(femalesTestTotal,1);
for i=1:femalesTestTotal
cfemalesMatrix2(i)=postDist(femalesTest(i,2:3),mean_females2,mean_males2,sigma_males2,sigma_females2,priorProb);
end


subplot(2,2,2);
scatter(malesTest(cmalesMatrix2(:,1)==1,2),malesTest(cmalesMatrix2(:,1)==1,3),'bx');
grid on;
hold on;
scatter(femalesTest(cfemalesMatrix2(:,1)==2,2),femalesTest(cfemalesMatrix2(:,1)==2,3),'ro');
hold off;
title('QDA with diagonal covariance');
xlabel('height');
ylabel('weight');
legend('Males', 'Females');

%Step 3: Calculate the accuracy
accuracy_males2 = sum(cmalesMatrix2(:)==1)/malesTestTotal;
accuracy_females2 = sum(cfemalesMatrix2(:)==2)/femalesTestTotal;

%% Classifier 3

%Step 1: fit the training data to the parameters. The parameters for the
%mean are the same as the previous classifier:
mean_males3 = meanMale;
mean_females3 = meanFemale;

%Calculate the covariance matrix by pooling the data from the males and
%females:
% Pool the data
trainingPooled = [femalesTrain(:,:);malesTrain(:,:)];
% Calculate the global mean height 
meanHeight = sum(trainingPooled(:,2))/(totalTrain);
% Calculate the global mean weight 
meanWeight = sum(trainingPooled(:,3))/(totalTrain);
% Vector containing the global means of heigh and wieght
meanGlobal = [meanHeight meanWeight];

runningMatrix = zeros(2,2); % Define a 2x2 empty matrix 
for i=1:totalTrain
    runningMatrix = runningMatrix + trainingPooled(i,2:3)'*trainingPooled(i,2:3);
end
sigma_global=(1/totalTrain)*runningMatrix - meanGlobal'*meanGlobal;

%Step 2: classify the test samples by calling the function with the new
%parameters
cmalesMatrix3 = zeros(malesTestTotal,1);
for i=1:malesTestTotal
cmalesMatrix3(i)=postDist(malesTest(i,2:3),mean_females3,mean_males3,sigma_global,sigma_global,priorProb);
end

cfemalesMatrix3 = zeros(femalesTestTotal,1);
for i=1:femalesTestTotal
cfemalesMatrix3(i)=postDist(femalesTest(i,2:3),mean_females3,mean_males3,sigma_global,sigma_global,priorProb);
end

%Step 3: Calculate the accuracy
accuracy_males3 = sum(cmalesMatrix3(:)==1)/malesTestTotal;
accuracy_females3 = sum(cfemalesMatrix3(:)==2)/femalesTestTotal; % Too good to be true?

subplot(2,2,3);
scatter(malesTest(cmalesMatrix3(:,1)==1,2),malesTest(cmalesMatrix3(:,1)==1,3),'bx');
grid on;
hold on;
scatter(femalesTest(cfemalesMatrix3(:,1)==2,2),femalesTest(cfemalesMatrix3(:,1)==2,3),'ro');
hold off;
title('LDA with shared covariance matrix');
xlabel('height');
ylabel('weight');
legend('Males', 'Females');

accuracy_total = [ accuracy_males,accuracy_females;accuracy_males2,accuracy_females2;accuracy_males3,accuracy_females3];


%% FUNCTIONS

function c= postDist(X,mean_females,mean_males,sigma_males,sigma_females,prior) % posterior distribution function

c_male = prior(1)*det(2*pi*sigma_males)^-0.5 * exp(-0.5 *(X-mean_males) * inv(sigma_males)*(X-mean_males)');

c_female = prior(2)*det(2*pi*sigma_females)^-0.5 * exp(-0.5 *(X-mean_females) * inv(sigma_females)*(X-mean_females)');

[~,c] = max([c_male/(c_male+c_female) c_female/(c_male+c_female)]);

end