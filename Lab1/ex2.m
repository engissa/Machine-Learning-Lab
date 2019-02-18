file_data = importdata('localization.mat'); %call for import the file name
train_raw = file_data.traindata(:,:,:); % call from KnnClassify2dTrian Table
test_raw = file_data.testdata(:,:,:); % call from KnnClassify2dTrian Table
train_data = zeros(120,8);
last_pos =0;
for i=1:24
    for j=1:5
    last_pos = last_pos +1;
    train_data(last_pos,:) = [train_raw(:,j:j,i)' i];  
    end
end
last_pos = 0;
test_data = zeros(120,8);
for i=1:24
    for j=1:5
    last_pos = last_pos +1;
    test_data(last_pos,:) = [test_raw(:,j:j,i)' i];  
    end
end
% mm_m = knn_predict(train_data,test_data,1:24,10);

K = 1:100;
errors_list_test = zeros(length(K),1);

for s = 1 : length(K)
k = K(s);
predicted_test = knn_predict(train_data,test_data,1:24,k);
testTotal = size(predicted_test,1);
errors_list_test(s,1) = (1-(1/testTotal)*sum(predicted_test(:,8)~=predicted_test(:,9)))*100;
%accuracy = (1 - errors)*100;
end
avg_error = zeros(length(K),1);
for s = 1 : length(K)
k = K(s);
predicted_train = knn_predict(train_data,test_data,1:24,k);
trainTotal = size(predicted_train,1);
av_errors=zeros(1,trainTotal);
for ff =1:trainTotal
    pred_value = predicted_train(ff,9);
    real_value = predicted_train(ff,8);
    pred_mat = [real_value,real_value-1,real_value+1,real_value-4,real_value+4];
    if ismember(pred_value,pred_mat)
        av_errors(ff)=1;
    end
end

avg_error(s,1) =(1/trainTotal)*sum(av_errors)*100;
end
figure
plot(K,errors_list_test(:,1),'-o',K,avg_error(:,1),'-o')
xlabel('K Value')
ylabel('Accuracy (%)')
legend('Top-k accuracy','Average accuracy')
