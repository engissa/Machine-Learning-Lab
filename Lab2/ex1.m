% LAB-2 Ex-1
% DATA Import
rawData=importdata('heightWeight.mat'); %call for import the file name
males = rawData(:,1)==1;

maleDataSet=rawData(males,:);
malesRaw = maleDataSet;
females = rawData(:,1)==2;
femaleDataSet=rawData(females,:);
malesTotal=size(maleDataSet,1);
femalesTotal=size(femaleDataSet,1);
% ---------------- % 

figure
subplot(2,3,1) 
% male Data in scatter plot
scatter(maleDataSet(:,2),maleDataSet(:,3));
xlabel('Height');
ylabel('Weight');
title('Males: H vs W');

%plot Height of Males as historgam
subplot(2,3,2);
hist(maleDataSet(:,2))
xlabel('Height');
ylabel('Occurences');
title('Males: Weight');
%plot Weight of Males as histogram
subplot(2,3,3);
hist(maleDataSet(:,3))
xlabel('Weight');
ylabel('Occurences');
title('Males: Weight');

%Female Data in scatter plot
subplot(2,3,4)
scatter(femaleDataSet(:,2),femaleDataSet(:,3));
xlabel('Height');
ylabel('Weight');
title('Females: H vs W');

%plot Height of Females as historgam
subplot(2,3,5);
hist(femaleDataSet(:,2))
xlabel('Height');
ylabel('Occurences');
title('Females: Height');
%plot Weight of Females as histogram
subplot(2,3,6);
hist(femaleDataSet(:,3))
xlabel('Weight');
ylabel('Occurences');
title('Females: Weight');

% Calculate the mean Vector of Males data set
% Vector contains the mean of Height and Weight
meanMale=mean(maleDataSet(:,2:3));
% Calculate the mean Vector of Females data set
% Vector contains the mean of Height and Weight
meanFemale=mean(femaleDataSet(:,2:3));

% maleDataSet(:,2:3) = maleDataSet(:,2:3) - meanMale;
% varMales = var(malesRaw(:,2:3));
% maleDataSet(:,2:3)=maleDataSet(:,2:3)./sqrt(varMales);

runningMatrix = zeros(2,2); % Define a 2x2 empty matrix 
for i=1:malesTotal
    runningMatrix = runningMatrix + maleDataSet(i,2:3)'*maleDataSet(i,2:3);
end

% Calculation of the Covariant Matrix of Males 
covMatrix_males=(1/malesTotal)*runningMatrix - meanMale'*meanMale;

runningMatrix = zeros(2,2);% Define a 2x2 empty matrix 
for i=1:femalesTotal
    runningMatrix = runningMatrix + femaleDataSet(i,2:3)'*femaleDataSet(i,2:3);
end

% Calculation of covarient Matrix of Females
covMatrix_females=(1/femalesTotal)*runningMatrix - meanFemale'*meanFemale;

% Plot the 2-D joint pdf for Males 
figure
x1 = 160:1:205; x2 = 50:1:130;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],meanMale,covMatrix_males);
F = reshape(F,length(x2),length(x1));
surf(x1,x2,F);
caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
% axis([160 205 50 230 0 max(F(:))])
xlabel('weight'); ylabel('height'); zlabel('Probability Density');
title('Males: Joint PDF');

% Plot the 2-D joint pdf for Females  

figure
x1 = 160:1:205; x2 = 50:1:130;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],meanFemale,covMatrix_females);
F = reshape(F,length(x2),length(x1));
surf(x1,x2,F);
caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
% axis([160 205 50 230 0 max(F(:))])
xlabel('weight'); ylabel('height'); zlabel('Probability Density');
title('Females: Joint PDF');