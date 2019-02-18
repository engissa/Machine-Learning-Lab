# Machine-Learning-Lab
Machine learning laboratory exercise for course Statistical Learning and Neural Networks

# LAB 1
## Exercise 1- synthetic dataset:
### Introduction:
In this exercise, we will use “synthentic.mat” dataset that contains two classes (1 and 2) and two features. This dataset is divided into two subsets: Training and Testing, each contains a list of examples. Our goal is to implement KNN-Classifier that predicts the classes of a new test set.
### Results:
![Graph](https://github.com/engissa/Machine-Learning-Lab/blob/master/Figures/knn1_fig2.png)
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
![Graph](https://github.com/engissa/Machine-Learning-Lab/blob/master/Figures/knn2_fig1.png)
### Discussion and Analysis:
The main challenge in this exercise is to convert the 3-dimensional array to a 2-dimensional matrix to be applied in the KNN classifier. After conversion, we end up with a 2d array of 120 training examples each has 7 features and a label that tells the room number. Classifier performance is measured depending on two factors: ​Top-K accuracy​ (the predicted class is the same as actual class) and ​Average accuracy​ (the predicted class is correct if it is one of the neighbors of the actual class). Top-K accuracy didn’t perform very well as it showed a maximum of 70.83% accuracy for k=2. On the other side, it showed an average accuracy 90% at most. This indicates that this classifier is experiencing ​curse of dimensionality​ as in the training dataset is highly dimensional (7 features/sensors) and 24 classes. Also, we can conclude from the average accuracy that as we decrease the number of classes (by looking also at neighbors) the accuracy increases by 20%. Finally, this classifier started to experience under fitting ​after K = 4 were both accuracy started to decrease and never showed improvement afterward.
## Exercise 3 - From classification to regression
### Introduction:
This exercise extends exercise 2 in the aspect of implementing KNN classifier that estimates the user spatial coordinates as a real-valued pair (x,y). The variable cell_coordinates in file localization.mat contains horizontal and vertical coordinates of the center of each cell in the first and second column respectively. The location of the user is estimated, for the horizontal and vertical coordinates, as a weighted mean of the coordinates of the k nearest-neighbor cells.
![Formula](https://github.com/engissa/Machine-Learning-Lab/blob/master/Figures/form1.png)
the sum is over the k nearest-neighbor cells, and 𝜉. is the horizontal or vertical coordinate of the center of cell i.
### Results:
![Formula](https://github.com/engissa/Machine-Learning-Lab/blob/master/Figures/knn3_fig1.png)
### Discussion and Analysis
Using KNN we were able to estimate the (x,y) coordinates of test examples. shows Euclidean distance of the prediction with the center position of the room. It is well visible that as we increase K the distance of the prediction coordinates gets more far away from the actual position. This might get us to realize and set a threshold of the distance away from the center of the room. Finally, it is interesting how KNN classifier can be modified to regress and predict continuous data instead of just classifying.