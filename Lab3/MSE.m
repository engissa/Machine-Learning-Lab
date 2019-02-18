function MSE = MSE(K, pooled_vector, Eigenvectors)
    
    pooled_vector_length1 = size(pooled_vector,1);

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

    MSE = 0;

    for i = 1:pooled_vector_length1
        %Note vectors are row vectors, so switch the transpose
        MSE = MSE + (pooled_vector(i,:) - pooled_vector_approximate(i,:)) * (pooled_vector(i,:) - pooled_vector_approximate(i,:))'; 
    end
    MSE = (1/(pooled_vector_length1)) * MSE;

end
