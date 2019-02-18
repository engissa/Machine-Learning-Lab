file_data = importdata('localization.mat'); %call for import the file name
train_raw = file_data.traindata(:,:,:); % call from KnnClassify2dTrian Table
test_raw = file_data.testdata(:,:,:); % call from KnnClassify2dTrian Table
cell_co = file_data.cell_coordinates(:,:);
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

K = 1:10;
errors_list_test = zeros(size(test_data,1),1);
test_res = zeros(120,1);
for s = 1 : length(K)
k = K(s);
predicted_test = knn_predict2(train_data,test_data,k,cell_co);
len_act= zeros(size(predicted_test,1),1);
for ff =1 : size(predicted_test)
len_act(ff)= pdist2(cell_co(predicted_test(ff,8),:),predicted_test(ff,9:10));
end
errors_list_test = [errors_list_test len_act];
% distt= pdist2(cell_co(nn_nearest(ff,1),:),predicted_test)
% errors_list_test = [errors_list_test predicted_test(:,9:10)];
% testTotal = size(predicted_test,1);
% errors_list_test(s,1) = (1/testTotal)*sum(predicted_test(:,8)~=predicted_test(:,9));
%accuracy = (1 - errors)*100;
end
% errors_list_train = zeros(length(K),1);
% 
% % error_k(k,1) = (1 - error_rate)*100;
% figure
%  plot(1:120,errors_list_test(:,2),'-x',1:120,errors_list_test(:,3),'-o',1:120,errors_list_test(:,10),'-o')
 bar(errors_list_test(:,5:8),'stacked')
 
% title('Knn Accuracy Vs K')
 xlabel('Example')
 ylabel('Distance')
 legend('k=4','k=5','k=6','k=7')
