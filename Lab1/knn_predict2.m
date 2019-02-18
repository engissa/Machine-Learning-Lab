function [ prediction_matrix ] = knn_predict2(train_Data,test_Data, K,cell_col )
cell_co = cell_col;
%   Detailed explanation goes here
testTotal = size(test_Data,1);
trainTotal = size(train_Data,1);
feat_cl = size(test_Data,2); %index of feature column
% Add two columns to test data to fill with probability and predicted class
test_data= [test_Data zeros(testTotal,2)];

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
   k_nearest = distances_matrix(1:K,1);
   nn_nearest = [k_nearest zeros(K,2)];
   for ff=1 : K
   nn_nearest(ff,2:3) = cell_co(nn_nearest(ff,1),:);
   end
   coord = sum(nn_nearest(:,2:3),1);
   test_data(test_row,9:10) = (1/K)*sum(coord,1);
end
prediction_matrix = test_data;
end