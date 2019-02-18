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

# LAB 2
## Exercise 1 - Model fitting for continuous distributions
### Introduction
In this exercise, we are using dataset (file heightWeigth.mat), containing labelled data for two classes, i.e. males and females. Our aim is to fit a class-conditional Gaussian multivariate distribution to these data, and visualize the probability density function Results
![Graphs](https://github.com/engissa/Machine-Learning-Lab/blob/master/Figures/lab2graphs.png)
## Discussion and Analysis
We can clearly visualize by the plotting of height and weight of each of the classes (males and females) that the features are not highly linear rather they are correlated (tall men/women have higher weight). On the other side, the histograms shows a clear Gaussian distribution fit on the random variables (height and weight). The Joint-probability plot of each class show that most of the data was represented in the pdf which makes it a good model for the current dataset classes.
## Exercise 2 – Classification – discrete data
### Introduction:
In this exercise, we are using dataset (XwindowsDocData.mat), that contains document features for two classes: Microsoft Windows and X Windows. The features are the presence/absence (1/0) of a word of the vocabulary in each document. The vocabulary has D=600 words. The training set contains the features for 900 documents (xtrain) as well as the specific class (ytrain). Our goal is to fit the parameters employed by a Naïve Bayes Classifier, using a Bernoulli model. Results:
![Graphs](https://github.com/engissa/Machine-Learning-Lab/blob/master/Figures/lab2ex2.png)
### Uninformative Features:
'exactly', 'remember', 'icons', 'forget', 'dragging', 'change', 'computer', 'engineering', 'introduced', 'alot', 'aspect','dedicated', 'cluster', 'addresses', 'winsock', 'disappear','double', 'fast', 'workstation', 'appears', 'paste', 'adress', 'possibly', 'circles', 'seperate', 'none', 'corner', 'stock','sense', 'characters', 'toolkits', 'bbs', 'responded', 'out','compuserve', 'per', 'pain', 'correct', 'utilities', 'components','day', 'involves', 'front', 'wife', 'march', 'duplicate', 'cpu','roberts', 'ignore', 'ease', 'copyright', 'looks', 'generator', 'vary', 'separated', 'recognition', 'leaving', 'beta', 'drives', 'grant', 'tseng', 'builtin', 'corrections', 'super', 'docs', 'phone', 'region', 'different', 'submitted', 'youve', 'patel', 'recall', 'eisa', 'just', 'gifs', 'forum', 'keys', 'drag', 'runs'
### Discussion:
In the above graphs the class conditional probability shows the most likely features to appear in each class. It is well visible that there is a spike of probability 1 on both classes for feature ‘dedicated’ that is completely uninformative in the Naive Bayes Classifier. To have a more visible visualization of the uninformative features the features of similar probabilities between the two classes were selected .It is clear that there is a high number of features that contain no information even though they have low probability. To conclude, this model can give a clear idea of which features to eliminate and which to focus on to build an efficient classifier.

## Exercise 3 - Classification – discrete data
### Introduction:
In this exercise, we want to use NBC classifier model implemented in exercise 2 to predict the classes of each Test vector in the test Dataset. The aim is to compare the accuracy of the classifier on the training and test data.
### Optional:
In this part, we will select the K most important features, and rerun the NBC classifier again to check how it is affecting the accuracy.
### Results:
#### Accuracy on All Features:
Test Dataset Accuracy : 81.4444 %
Train Dataset Accuracy : 91.6667 %
Accuracy Difference : 10.22 %
![Graphs](https://github.com/engissa/Machine-Learning-Lab/blob/master/Figures/ex3_fig4o.png)
### Discussion and Analysis:
The NBC model built in exercise 2 was used to predict the class of the test and training set using the MAP estimate. The classifier showed an accuracy of 81.4% on the test data which might be low due to overfitting of un-informative features. Moreover, the training set shows an accuracy of 91.6% after testing on trained data of itself shows that the model is not memory-based and it can perform a bit better than test data however not to an intent of giving 100% accuracy.
To better visualize how choosing the correct features for the model, features were sorted by importance (how much information they carry) and K features were selected. The above graph shows that even if one important feature was selected, it wasn’t enough to raise the accuracy of the classifier. As the number of selected features increased, the accuracy improved to around 85.5% at most and after that is started to decrease back to 81%. To conclude, feature information is critical to the model classifier performance as it can show a big improvement if just the most important features are selected.

## Exercise 4 – Classification: Continuous Data
### Introduction:
In this exercise, the dataset (file heightWeigth.mat) is used to build a classifier using Gaussian Discriminative analysis. The accuracy of the classifier is computed across several versions of the classifier:
1) Two-class quadratic discriminant analysis (fit mean and covariance are class specific) 2) Two-class quadratic discriminant analysis with diagonal covariance matrices
3) Two class linear discriminant analysis (fit a shared covariance matrix)
### Results:
![Graphs](https://github.com/engissa/Machine-Learning-Lab/blob/master/Figures/l2ex4.png)
 Classifier| Males (%) | Females (%) | Average (%)
 --------- | --------- | ----------- | -------
Quadratic Discriminant Analysis | 84 | 97.14 | 90.57
QDA with diagonal covariance | 83 | 91 | 87
LDA with shared covariance matrix | 60 | 100 | 80

### Discussion:
150 training samples consisted of 50 males and 100 females with unequal prior probability of 0.33 and 0.67 respectively.
QDA resulted in an quadratic decision boundary that is good for non-linearly separable data.. On the other hand , QDA with diagonal covariance matrix assumes the off-diagonal elements of covariance matrix to be zero which also had a quadratic boundary but was less biased to females. LDA shares the covariance matrix across both classes that turns the decision surface to linear .

# Lab 3
## Exercise 1 – PCA
### Introduction:
In this exercise, a dataset of two classes “10: Soybean-notill” and “2: Corn-notill” was chosen from “Indian_Pines” dataset. Each example has a large number of features(220) in each so our goal is to reduce the dimensionality by implementing PCA and analyzing how it affects cost function(MSE).
### Results:
![Graphs](https://github.com/engissa/Machine-Learning-Lab/blob/master/Figures/Lab31.png)
### Discussion
PCA was implemented to reduce the features dimensionality of our dataset from 220 to a new dataset of K uncorrelated significant features (principal components). PCA implies the distribution centering by subtracting the mean. Afterward, the dataset was divided into two sets Train and Test. MSE was calculated on different number of principal components to interpret how choosing K components is affecting the error rate of the classifier. It is clearly in the above figure the increase of principal components in the newly produced dataset increases accuracy of the classifier. In addition, after taking the first 100 components the error rate of the classifier became constant and didn’t show any change for any additional components. Hence, choosing only first K components will be enough to reconstruct the dataset that satisfy the computational capabilities.
### Additional:
The same testing was done on normalized data , that showed how the MSE highly decreased for lower values of K.
