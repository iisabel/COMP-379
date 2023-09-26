#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 16:52:18 2022

@author: isabelheard
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
import numpy.random
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns





#Question 1 & 2
#Percepron algorithm
class Perceptron(object):
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state


    def inputs(self, X):

        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):

        return np.where(self.inputs(X) >= 0.0, 1, -1)
    
    
    def fit(self, X, y):
        
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    
#Make the data sets
def make_data(n):
    Y = np.random.choice([-1, +1], size=n)
    X = np.random.normal(size = (n, 2))
    for i in range(len(Y)):
        X[i] += Y[i]*np.array([-2, 0.9])
    return X, Y


#Linearly Seperable Data set
X, Y = make_data(20)

plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=Y, cmap='Paired_r', edgecolors='c')

perc = Perceptron()
perc.fit(X,Y)

#plot decision bondary
plot_decision_boundary(perc, X, Y)


#Plot linear model 
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, Y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()




#Non linearly Seperable
A, B = make_data(100)

plt.figure(figsize=(5,5))
plt.scatter(A[:,0], A[:,1], c=B, cmap='Paired_r', edgecolors='c')

perc = Perceptron()
perc.fit(A,B)
#plot decision bondary
plot_decision_boundary(perc, A, B)

#Plot non linear model
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(A, B)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()




#Questions 3
#read in training data
train = pd.read_csv('train.csv')
train.head()


#read in test data
test = pd.read_csv('test.csv')
test.head()
test.info()

#See how many rows an columns there are
print(train)

#View the column values
print(train.columns.values)

#Finding different data types
train.info()


#DATA CLEANING

#Drop uneccesary info
train_df = train.drop(['PassengerId', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Name'], axis = 1)
test_df = test.drop(['Ticket', 'Fare', 'Cabin', 'Embarked', 'Name'], axis = 1)

#Fixing null values with averages or modes
train_df["Age"] = train_df["Age"].fillna(train_df['Age'].median())
test_df["Age"] = test_df["Age"].fillna(test_df['Age'].median())



#Convert the floats to ints
train_df.Age = train_df.Age.astype(int)
test_df.Age = test_df.Age.astype(int)

#dummy variables for sex
sex_dummy_train = pd.get_dummies(train['Sex'])
sex_dummy_test = pd.get_dummies(test['Sex'])

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


#Spliting the data
x_train, y_train= train_test_split(train_df, test_size=0.3, random_state=25)
print(f"No. of training examples: {x_train.shape[0]}")  #623
print(f"No. of testing examples: {y_train.shape[0]}")   #268


class AdalineGD(object):

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)

            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


#define true class label & training data
y = train_df.iloc[0:891, 0].values
y = np.where(y == 1, 1, -1)

x = train_df.iloc[0:891, [1,8]].values

#read test data
x_t = test_df.iloc[0:418, [1,8]].values


#train with adaline
ada_tit = AdalineGD(n_iter = 20, eta = 0.0001).fit(x, y)

#plot the training cost
plt.plot(range(1, len(ada_tit.cost_) + 1), ada_tit.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')

plt.tight_layout()
plt.show()



#Prediction for tests
y_predict = ada_tit.predict(x_t)
y_predict




#Question 4
train_df.corr()["Survived"]

train_corr = train_df.drop(columns=[]).corr(method='pearson')
plt.figure(figsize=(18, 12))
sns.set(font_scale=1.4)
sns.heatmap(train_corr, 
            annot=True, 
            linecolor='white', 
            linewidth=0.5, 
            cmap='magma');



#Question 5
#Baseline model
XX, YY = make_data(100)

plt.figure(figsize=(5,5))
plt.scatter(XX[:,0], XX[:,1], c=YY, cmap='Paired_r', edgecolors='c')


#Baseline for Adaline
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ada1 = AdalineGD(n_iter=10, eta=0.01).fit(XX, YY)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = AdalineGD(n_iter=10, eta=0.001).fit(XX, YY)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.001')

plt.show()


#Baseline for Perceptron
perc = Perceptron()
perc.fit(XX,YY)

#plot decision bondary
plot_decision_boundary(perc, XX, YY)

