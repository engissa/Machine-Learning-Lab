%% Exercise 2

%Load the data
load('Indian_Pines_Dataset')

%Step 1: Plot the mean vector of class 0 and 1
%Extract vectors belonging to class 0 or 1.
size_class0 = 1265;
size_class1 = 1428;
class0 = zeros(size_class0,220);
class1 = zeros(size_class1,220);
n0 = 0;
n1 = 0;
for i=1:size(indian_pines,1)
    for j=1:size(indian_pines,2)
        if indian_pines_gt(i,j)== 10 % class index
        n0=n0+1;
        class0(n0,:)= indian_pines(i,j,:);
        end       
        if indian_pines_gt(i,j)== 2 % class index
        n1=n1+1;
        class1(n1,:)= indian_pines(i,j,:);
        end      
    end
end

%Divide the data into training and test data
class0_train = class0(1:round(0.75*size_class0),:);
class0_test = class0( (round(0.75*size_class0)+1):size_class0,:);
class1_train = class1(1:round(0.75*size_class1),:);
class1_test = class1( (round(0.75*size_class1)+1):size_class1,:);


%Subtract the mean values of the sample vectors, to get a zero-mean vector
%Compute the mean of the whole dataset, for each feature
class0n_train = class0_train - mean(class0_train);
class1n_train = class1_train - mean(class1_train);

plot(mean(class0))
hold on
plot(mean(class1))
legend('Class 1' , 'Class 2')
xlabel('Feature index')
ylabel('Mean value')
hold off

%Step 2: Apply the classifier to the original data (without PCA) and compute its accuracy
x0 = 0.5*(mean(class0_train) + mean(class0_train));
w = mean(class0_train) - mean(class1_train);

num_error = 0;
for i = 1:size(class0_test,1)
    num_error = num_error + 1*( sign(w*(class0_test(i,:)-x0)') == -1 ); %Note vectors are row vectors, so switch the transpose
end
for i = 1:size(class1_test,1)
    num_error = num_error + 1*( sign(w*(class1_test(i,:)-x0)') == 1 ); %Note vectors are row vectors, so switch the transpose
end
accuracy = 1 - num_error / (size(class0_test,1) + size(class1_test,1)); %Fraction of correctly classified values
