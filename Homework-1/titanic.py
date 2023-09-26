#DATA IMPORTING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#read in training data
train = pd.read_csv('train.csv')
train.head()

#read in test data
test = pd.read_csv('test.csv')
test.head()

#See how many rows an columns there are
print(train)

#View the column values
print(train.columns.values)

#Finding different data types
train.info()











#DATA CLEANING

#Seeing how many null values there are
train.isnull().sum() #Age, Cabin, Embarked
test.isnull().sum()  #Age, Cabin, Fare


#Fixing null values with averages or modes
train["Age"] = train["Age"].fillna(train['Age'].median())
test["Age"] = test["Age"].fillna(test['Age'].median())
train["Embarked"] = train["Embarked"].fillna(train['Embarked'].mode()[0])
test["Fare"] = test["Fare"].fillna(test['Fare'].median())


#Convert the floats to ints
train.Age = train.Age.astype(int)
test.Age = test.Age.astype(int)
train.Fare = train.Fare.astype(int)
test.Fare = test.Fare.astype(int)









#DATA ANALYZING

#Looking at Survivors (1) vs non survivors (0)
train.groupby('Survived').Survived.count().plot.bar(ylim=0)
plt.show()



#Seeing what sex had a higher survival rate
sexVar = {'male':1, 'female':2}
train.Sex = train.Sex.map(sexVar)
sexVar2 = train.groupby('Sex')['Survived'].sum().reset_index()
sexVar2    #More woman survived than men



#Survival Rate by Pclass (1 = upper class, 2 = middle class, 3 = lower class)
class1_SR = np.sum((train.Pclass == 1) & (train.Survived == 1)) / np.sum(train.Pclass == 1)
class2_SR = np.sum((train.Pclass == 2) & (train.Survived == 1)) / np.sum(train.Pclass == 2)
class3_SR = np.sum((train.Pclass == 3) & (train.Survived == 1)) / np.sum(train.Pclass == 3)
class_SR = np.array([class1_SR, class2_SR, class3_SR])
class_DR = 1 - class_SR

plt.figure(figsize=[6,4])
plt.bar(['Class 1', 'Class 2', 'Class 3'], class_SR, label='Survived', 
        color='Blue', edgecolor='k')
plt.bar(['Class 1', 'Class 2', 'Class 3'], class_DR, label='Died', 
        bottom=class_SR, color='Red', edgecolor='k')
plt.legend(loc="center left", bbox_to_anchor=(1.03,0.5))
plt.ylabel('Proportion')
plt.title('Survival Rate by Class')
plt.show()
#The higher the class, the more likely you are to survive



#Survival Rate by age group
plt.figure(figsize=[8,6])
plt.hist([train.Age.values[train.Survived == 1], train.Age.values[train.Survived == 0]], 
         bins=np.arange(0,90,5), label=['Survived','Died'], density=True,
         edgecolor='k', alpha=0.6, color=['Blue','Red'])
plt.xticks(np.arange(0,90,5))
plt.legend()
plt.xlabel('Age')
plt.ylabel('Proportion')
plt.title('Survival Rates by Age Group')
plt.show()
#The younger you were, the better change you had at surviving


#Correlation matrix between survival rate and variables
train.corr()["Survived"]

train_corr = train.drop(columns=['PassengerId']).corr(method='pearson')
plt.figure(figsize=(18, 12))
sns.set(font_scale=1.4)
sns.heatmap(train_corr, 
            annot=True, 
            linecolor='white', 
            linewidth=0.5, 
            cmap='magma');
#Pclass, Age, SibSb all had a negative correlation with surviving
#Sex, Parch, and Fare all have a positive correlation with surviving
#Cabin, Ticket, Name, Fare, and PassengerID do not seem needed in this analysis


#New data set?
df = pd.read_csv("train.csv", usecols= ["PassengerId", "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]) 
df.head() 



#Survived vs important variables

#First class saw the most survivors, while third class saw the most deaths
print(pd.pivot_table(train, index = 'Survived', columns = 'Pclass',values = 'Ticket' ,aggfunc ='count'))
print()

#The majority of woman survived
print(pd.pivot_table(train, index = 'Survived', columns = 'Sex', values = 'Ticket' ,aggfunc ='count'))
print()

#This comparison does not seem relevant, but I guess if you left from Cherbourg you had a better chance of surviving
print(pd.pivot_table(train, index = 'Survived', columns = 'Embarked', values = 'Ticket' ,aggfunc ='count'))
print()

#Does not really seem to matter if you had a sibiling or spouse
print(pd.pivot_table(train, index = 'Survived', columns = 'SibSp', values = 'Ticket' ,aggfunc ='count'))
print()

#No real difference
print(pd.pivot_table(train, index = 'Survived', columns = 'Parch', values = 'Ticket' ,aggfunc ='count'))
print()

    


#Calculating survival rates

#First class Children survival rate
first_children = train[(train['Age'] < 18) & (train['Pclass'] == 1)]
first_c_survival = first_children['Survived'].value_counts(normalize=True) * 100
first_c_survival
#1    91.666667
#0     8.333333


#Third class Children survival rate
third_children = train[(train['Age'] < 18) & (train['Pclass'] == 3)]
third_c_survival = third_children['Survived'].value_counts(normalize=True) * 100
third_c_survival
#0    62.820513
#1    37.179487


#third class man survival rate
third_man = train[(df['Sex'] == "male") & (train['Age'] < 18) & (train['Pclass'] == 3)]
third_man_SR = third_man['Survived'].value_counts(normalize=True) * 100
third_man_SR
#0    76.744186
#1    23.255814


#first class man survival rate
first_man = train[(df['Sex'] == "male") & (train['Age'] < 18) & (train['Pclass'] == 1)]
first_man_SR = first_man['Survived'].value_counts(normalize=True) * 100
first_man_SR
#0    64.705882
#1    35.294118


#Third class women survival rate
third_woman = train[(df['Sex'] == "female") & (train['Age'] < 18) & (train['Pclass'] == 3)]
third_woman_SR= third_woman['Survived'].value_counts(normalize=True) * 100
third_woman_SR
#1    54.285714
#0    45.714286


#First class women survival rate
first_woman = train[(df['Sex'] == "female") & (train['Age'] < 18) & (train['Pclass'] == 1)]
first_woman_SR = first_woman['Survived'].value_counts(normalize=True) * 100
first_woman_SR
#1    87.5
#0    12.5



#Notes

#If you are a:
    #Sex - female 
    #Age - young  
    #pclass - Upper class 
    #Embarked - if you embarked from Cherbourg
#You have a better chance of surviving