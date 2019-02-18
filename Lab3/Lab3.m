%% Exercise 1

%Load the data
load('Indian_Pines_Dataset')

%Step 1: Extract vectors belonging to class 1 or 2.
size_class1 = 1265;
size_class2 = 1428;
class1 = zeros(size_class1,220);
class2 = zeros(size_class2,220);
n1 = 0;
n2 = 0;
for i=1:size(indian_pines,1)
    for j=1:size(indian_pines,2)
        if indian_pines_gt(i,j)== 14 % class index
        n1=n1+1;
        class1(n1,:)= indian_pines(i,j,:);
        end       
        if indian_pines_gt(i,j)== 2 % class index
        n2=n2+1;
        class2(n2,:)= indian_pines(i,j,:);
        end      
    end
end

%Subtract the mean values of the sample vectors, to get a zero-mean vector
%Compute the mean of the whole dataset, for each feature
class1n = class1 - mean(class1);
class2n = class2 - mean(class2);
pooled_vector = [class1n ; class2n];
mean_vector = mean(pooled_vector);
pooled_vector = pooled_vector./std(pooled_vector); % Standardizing the data

%Step 2: Compute the sample covariance matrix
pooled_vector_length1 = size(pooled_vector,1); %Number of samples
pooled_vector_length2 = size(pooled_vector,2); %Number of features of each sample

sample_covariance = zeros(pooled_vector_length2 , pooled_vector_length2); %

for i = 1:pooled_vector_length1
    sample_covariance = sample_covariance + pooled_vector(i,:)' * pooled_vector(i,:); %Note vectors are row vectors, so switch the transpose
end
sample_covariance = (1/(pooled_vector_length1)) * sample_covariance;

%Step 3: Compute the eigenvalues and eigenvectors of the sample covariance
%matrix
[Eigenvectors , Eigenvalues] = eig(sample_covariance);

%Step 4: Pick a number of dimensions K <= 220
K = 150;

%Step 5: Construct the eigenvector matrix W for K components (i.e., select the last K columns)
sorted_Eigenvectors = fliplr(Eigenvectors);
largest_Eigenvectors = sorted_Eigenvectors(:,1:K);

%Step 6: Using W, compute the PCA coefficients for each spectral vector in the test set
z = zeros(pooled_vector_length1,K);
for i = 1:pooled_vector_length1
    z(i,:) = largest_Eigenvectors' * pooled_vector(i,:)'; %Note, the pooled_vector is a row vector, so we need to transpose
end

%Step 7: Then from the PCA coefficients obtain an approximation of the corresponding test vector
% and compute the error (mean square error - MSE)
pooled_vector_approximate = largest_Eigenvectors * z';
pooled_vector_approximate = pooled_vector_approximate'; %Transpose for convenience

MSE1 = 0;
for i = 1:pooled_vector_length1
    %Note vectors are row vectors, so switch the transpose
    MSE1 = MSE1 + (pooled_vector(i,:) - pooled_vector_approximate(i,:)) * (pooled_vector(i,:) - pooled_vector_approximate(i,:))'; 
end
MSE1 = (1/(pooled_vector_length1)) * MSE1;

%Step 8: Plot the average MSE over the test set as a function of K.
%We construct a function MSE that executes step 5 to 7 for different values
%of K.
MSE_k = zeros(pooled_vector_length2,1);
for i = 1:pooled_vector_length2
    MSE_k(i) = MSE(i, pooled_vector, Eigenvectors);
end

%% Plot exercise 1
plot(MSE_k)
title('Standardized Data - MSE value for different values of K')
xlabel('K')
ylabel('MSE')