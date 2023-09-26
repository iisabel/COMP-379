#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 11:35:27 2022

@author: isabelheard
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from random import randrange

#Used for checking work
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold as kfoldC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV as Grid

#read in training data
train = pd.read_csv('train.csv')
train.info()

#read in test data
test = pd.read_csv('test.csv')
test.info()

#Look at nulls
train.isnull().sum()  #Age, Cabin, Embarked
test.isnull().sum()   #Age, Fare, Cabin 

#DATA CLEANING
#Drop variables with a lot of missing values, or no correlation to predictor variable
train_df = train.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], axis = 1)
test_df = test.drop(['Ticket', 'Cabin', 'Name'], axis = 1)

#Fixing null values with the mean
train_df["Age"] = train_df["Age"].fillna(train_df['Age'].mean())
test_df["Age"] = test_df["Age"].fillna(test_df['Age'].mean())

test_df["Fare"] = test_df["Fare"].fillna(test_df['Fare'].mean())

#Filling in Na values with most common value
train_df['Embarked'] = train_df['Embarked'].fillna('S')

#Convert the floats to ints
train_df.Age = train_df.Age.astype(int)
test_df.Age = test_df.Age.astype(int)

train_df.Fare = train_df.Fare.astype(int)
test_df.Fare = test_df.Fare.astype(int)

train_df['Embarked'] = pd.factorize(train_df['Embarked'])[0]
test_df['Embarked'] = pd.factorize(test_df['Embarked'])[0]

#dummy variables for sex
sex_dummy_train = pd.get_dummies(train_df['Sex'])
sex_dummy_test = pd.get_dummies(test_df['Sex'])

train_df = train_df.join(sex_dummy_train)
test_df = test_df.join(sex_dummy_test)

train_df = train_df.drop(['Sex'], axis = 1)
test_df = test_df.drop(['Sex'], axis = 1)

#dummy variables for Pclass
pc_dummy_tit = pd.get_dummies(train_df['Pclass'])
pc_dummy_test = pd.get_dummies(test_df['Pclass'])

pc_dummy_tit.columns = ['Class_1', 'Class_2', 'Class_3']
pc_dummy_test.columns = ['Class_1', 'Class_2', 'Class_3']

train_df = train_df.join(pc_dummy_tit)
test_df = test_df.join(pc_dummy_test)


train_df.isnull().sum() 
test_df.isnull().sum() 

print(train_df.head())
print(test_df.head())

train_df.info()
test_df.info()




#Problem 1
y = train_df.Survived
x = train_df.drop("Survived", axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=10)

print("shape of original dataset :", train_df.shape)
print("shape of input - training set", x_train.shape)
print("shape of output - training set", y_train.shape)
print("shape of input - testing set", x_test.shape)
print("shape of output - testing set", y_test.shape)
#Best performence metric - accuracy score



#Problem 2
#K-fold CV from scratch & use classifier of choice (SVM or logistic regression)   

class kFoldCV:
    def __init__(self):
        pass
        
    def printMetrics(self, actual, predictions):
        assert len(actual) == len(predictions)
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predictions[i]:
                correct += 1
        return (correct / float(len(actual)) * 100.0)

    
    def crossValSplit(self, dataset, numFolds):
        dataSplit = list()
        dataCopy = list(dataset)
        foldSize = int(len(dataset) / numFolds)
        for _ in range(numFolds):
            fold = list()
            while len(fold) < foldSize:
                index = randrange(len(dataCopy))
                fold.append(dataCopy.pop(index))
            dataSplit.append(fold)
        return dataSplit
    
    
    def kFCVEvaluate(self, dataset, numFolds, *args):
        #knn = kNNClassifier()
        folds = self.crossValSplit(dataset, numFolds)
        print("\nDistance Metric: ",*args[-1])
        print('\n')
        scores = list()
        for fold in folds:
            trainSet = list(folds)
            trainSet.remove(fold)
            trainSet = sum(trainSet, [])
            testSet = list()
            for row in fold:
                rowCopy = list(row)
                testSet.append(rowCopy)
                
            trainLabels = [row[-1] for row in trainSet]
            trainSet = [train[:-1] for train in trainSet]
            knn.fit(trainSet,trainLabels)
            
            actual = [row[-1] for row in testSet]
            testSet = [test[:-1] for test in testSet]
            
            predicted = knn.predict(testSet, *args)
            
            accuracy = self.printMetrics(actual, predicted)
            scores.append(accuracy)
        print('*'*20)
        print('Scores: %s' % scores)
        print('*'*20)
        print('\nMaximum Accuracy: %3f%%' % max(scores))
        print('\nMean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))


#Check with scikit
# range of k we want to try
k_range = range(1, 30)
# empty list to store scores
k_scores = []

#loop through reasonable values of k
for k in k_range:
    #run KNeighborsClassifier with k neighbours
    knn = KNeighborsClassifier(n_neighbors=k)
    #obtain cross_val_score for KNeighborsClassifier with k neighbours
    scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
    #append mean of scores for k neighbors to k_scores list
    k_scores.append(scores.mean())
print(k_scores)
print('Max of list', max(k_scores))

# plot the value of K (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K')
plt.ylabel('Cross-validated accuracy')

#Logistic regression
kfold = kfoldC(n_splits=10, random_state=0, shuffle=True)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, x_train, y_train, cv=kfold)
print("Accuracy: %.3f%%" % (results.mean()*100.0))


 
#Problem 3
#Grid search from scratch, include 2 hyper-parameters
class LogitRegression() :
	def __init__( self, learning_rate, iterations ) :		
		self.learning_rate = learning_rate		
		self.iterations = iterations
		
	# Function for model training			
	def fit( self, X, Y ) :		
		# no_of_training_examples, no_of_features		
		self.m, self.n = X.shape
		
		# weight initialization		
		self.W = np.zeros( self.n )		
		self.b = 0		
		self.X = X		
		self.Y = Y
		
		# gradient descent learning				
		for i in range( self.iterations ) :			
			self.update_weights()			
		return self
	
	# Helper function to update weights in gradient descent	
	def update_weights( self ) :		
		A = 1 / ( 1 + np.exp( - ( self.X.dot( self.W ) + self.b ) ) )
		
		# calculate gradients		
		tmp = ( A - self.Y.T )		
		tmp = np.reshape( tmp, self.m )		
		dW = np.dot( self.X.T, tmp ) / self.m		
		db = np.sum( tmp ) / self.m
		
		# update weights	
		self.W = self.W - self.learning_rate * dW	
		self.b = self.b - self.learning_rate * db		
		return self
	
	# Hypothetical function h( x )	
	def predict( self, X ) :	
		Z = 1 / ( 1 + np.exp( - ( X.dot( self.W ) + self.b ) ) )		
		Y = np.where( Z > 0.5, 1, 0 )		
		return Y
		
	

def grid() :	
	max_accuracy = 0
	
	# learning_rate choices	
	learning_rates = [ 0.1, 0.2, 0.3, 0.4, 0.5,]
	
	# iterations choices	
	iterations = [ 100, 200, 300, 400, 500 ]
	
	parameters = []	
	for i in learning_rates :		
		for j in iterations :			
			parameters.append( ( i, j ) )
			
	print("Available combinations : ", parameters )
			
	# Applying linear searching in list of available combination
	# to achieved maximum accuracy on CV set
	
	for k in range( len( parameters ) ) :		
		model = LogitRegression( learning_rate = parameters[k][0],
								iterations = parameters[k][1] )
	
		model.fit( x_train, y_train )
		
		# Prediction on validation set
		y_pred = model.predict( x_test )
	
		# measure performance on validation set
	
		correctly_classified = 0
	
		# counter	
		count = 0
	
		for count in range( np.size( y_pred ) ) :			
			if y_test[count] == y_pred[count] :				
				correctly_classified = correctly_classified + 1
				
		curr_accuracy = ( correctly_classified / count ) * 100
				
		if max_accuracy < curr_accuracy :			
			max_accuracy = curr_accuracy
			
	print( "Maximum accuracy achieved by our model through grid searching : ", max_accuracy )



#Scikit learn check grid search cross validation Logistic Regression
grids={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
logreg=LogisticRegression()
logreg_cv=Grid(logreg,grids,cv=10)
logreg_cv.fit(x_train,y_train)
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)
logreg2=LogisticRegression(C=10,penalty="l2")
logreg2.fit(x_train,y_train)
print("score",logreg2.score(x_train,y_train)) #80.76%


#Problem 4
#Evaluate the best model and report performance on test data
logreg2=LogisticRegression(C=10,penalty="l2")
logreg2.fit(x_test,y_test)
print("score",logreg2.score(x_test,y_test)) #82.68




















