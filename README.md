# Machine-Learning-Lab
Machine learning laboratory exercise for course Statistical Learning and Neural Networks

# LAB 1
## Exercise 1- synthetic dataset:
### Introduction:
In this exercise, we will use “synthentic.mat” dataset that contains two classes (1 and 2) and two features. This dataset is divided into two subsets: Training and Testing, each contains a list of examples. Our goal is to implement KNN-Classifier that predicts the classes of a new test set.
### Results:
![Image of Yaktocat](https://github.com/engissa/Machine-Learning-Lab/knn1_fig2.png)
### Discussion and Analysis:
The maximum accuracy achieved on Train Set was for ​k​=1was 100% while the maximum Accuracy on Test for K=5 was 97%. However even though the accuracy for train set was 100% for k=1, it wasn’t the case for the Test Set on k=1 which showed much lower value 90%. Therefore k=1 may not work well at predicting future data and the model would be very complex and experience ​over fitting​. We can also see that using K = 5 results in a better prediction on test data, because we are averaging over a larger neighborhood. As K increases, the predictions become smoother until k =25 where we started to see decrease in accuracy on both sets. This is where the model becomes very simple it experiences ​under fitting​.
Finally, whenever the classifier is stuck between two classes that have equal probabilities we decided to choose randomly between the two classes. However, the real state of the classification is ​undecided.
### Additional Thoughts:
Instead of deciding each time we want to classify data on the K values, it would be better if we average the predicted classes over a set of K values. This makes the classifier more robust and stable on getting better predictions.
Moreover, KNN classifier might be good classifier for this dataset because it has low dimensionality and was able to achieve an accuracy of 97% with just few training examples.
## Exercise - 2
### Introduction:
In this exercise, we are using “localization.mat” dataset containing two variables, called traindata and testdata. These variables have the same size, and are 3- dimensional arrays of size D=7, M=5, and 𝑁" = 24. Our goal is to implement a k-NN classifier for the classifying the cell in which the user is located in, and evaluate its performance.
### Results:
