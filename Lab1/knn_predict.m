function [ prediction_matrix ] = knn_predict(train_Data,test_Data,Classes, K )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
testTotal = size(test_Data,1);
trainTotal = size(train_Data,1);
feat_cl = size(test_Data,2); %index of feature column
% Add two columns to test data to fill with probability and predicted class
test_data= [test_Data zeros(testTotal,2)];
C= length(Classes); % Contains the number of classes 
% Alternate trough each Xi of the test set and distances Matrix for that
% element. Each Row in that matrix contains the distance between Xi(test) and
% Xj(train) along with the class on train

for test_row=1 : testTotal

    distances_matrix = zeros(trainTotal,2);
    
    for train_row=1 : trainTotal
        distances_matrix(train_row,1) = train_Data(train_row,feat_cl); % get the class
        % Calculate the distance between test X vector and train X vector
        distances_matrix(train_row,2) = pdist2(test_data(test_row,1:feat_cl-1),train_Data(train_row,1:feat_cl-1));
    end
   % after calculating all the distances we need to sort by shortest
   % distance (ascending)
   distances_matrix = sortrows(distances_matrix,2);
   % we select the  K nearest elements from distances_matrix
   k_nearest = distances_matrix(1:K,:);
   res = [Classes' zeros(C,1)]; % list containing the classes and the number of occurencies 
   
   for m=1 : C 
   res(m,2)= (1/K)*sum(k_nearest(:,1) == res(m,1)); % go through all classes and check prediction
   end
   
   [M,I] = max(res(:,2)); % get maximum of predicted class 
   MM = find(res(:,2) == max(res(:,2))); % used to check if there is any equal probabilities
   if length(MM)>1
     test_data(test_row,feat_cl+1)=res(MM(randi(length(M))),1); % if there are equal probabilties choos randomly between them
%     test_data(test_row,feat_cl+1) = -10;
   else
   test_data(test_row,feat_cl+1) = res(I,1);
   end
end
prediction_matrix = test_data;
end