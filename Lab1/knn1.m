file_data = importdata('synthetic.mat'); %call for import the file name
% Get all Training Data
train_raw = file_data.knnClassify2dTrain(:,:); % call from KnnClassify2dTrian Table
gscatter(train_raw(:,1),train_raw(:,2),train_raw(:,3))

% Get all Testing Data
test_raw=file_data.knnClassify2dTest(:,:);
  K = [1 3 5 12 16 20 25 35];
% K = 1:35;
errors_list_test = zeros(length(K),1);

for s = 1 : length(K)
k = K(s);
predicted_test = knn_predict(train_raw,test_raw,1:2,k);
testTotal = size(predicted_test,1);
errors_list_test(s,1) = (1-(1/testTotal)*sum(predicted_test(:,3)~=predicted_test(:,4)))*100;
%accuracy = (1 - errors)*100;
end
errors_list_train = zeros(length(K),1);

for s = 1 : length(K)
k = K(s);
predicted_train = knn_predict(train_raw,train_raw,1:2,k);
trainTotal = size(predicted_train,1);
errors_list_train(s,1) =(1-(1/trainTotal)*sum(predicted_train(:,3)~=predicted_train(:,4)))*100;
%accuracy = (1 - errors)*100;
end
% error_k(k,1) = (1 - error_rate)*100;
figure
plot(K,errors_list_test(:,1),K,errors_list_train(:,1))
title('Knn Accuracy Vs K')
xlabel('K Value')
ylabel('Accuracy (%)')
legend('Test','Train')