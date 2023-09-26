import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#pre-processing
df = pd.read_csv('weatherAUS.csv')

#looking at the statistics of each variable
df.describe()
df.head()

#View the column values
print(df.columns.values)

#Finding different data types
df.info()

#look at null values
df.isnull().sum() 

#percent of null missing
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
print(missing_value_df)

#Filling averages for missing some missing variables 
df["MinTemp"] = df["MinTemp"].fillna(df['MinTemp'].mean())
df["MaxTemp"] = df["MaxTemp"].fillna(df['MaxTemp'].mean())
df["Rainfall"] = df["Rainfall"].fillna(df['Rainfall'].mean())
df["Humidity9am"] = df["Humidity9am"].fillna(df['Humidity9am'].mean())
df["Humidity3pm"] = df["Humidity3pm"].fillna(df['Humidity3pm'].mean())
df["Pressure9am"] = df["Pressure9am"].fillna(df['Pressure9am'].mean())
df["Pressure3pm"] = df["Pressure3pm"].fillna(df['Pressure3pm'].mean())
df["Temp9am"] = df["Temp9am"].fillna(df['Temp9am'].mean())
df["Temp3pm"] = df["Temp3pm"].fillna(df['Temp3pm'].mean())
df["WindGustSpeed"] = df["WindGustSpeed"].fillna(df['WindGustSpeed'].mean())
df["WindSpeed9am"] = df["WindSpeed9am"].fillna(df['WindSpeed9am'].mean())
df["WindSpeed3pm"] = df["WindSpeed3pm"].fillna(df['WindSpeed3pm'].mean())

#Filling the missing values for continuous variables with mode
df['RainToday']=df['RainToday'].fillna(df['RainToday'].mode()[0])
df['RainTomorrow']=df['RainTomorrow'].fillna(df['RainTomorrow'].mode()[0])
df['WindDir9am'] = df['WindDir9am'].fillna(df['WindDir9am'].mode()[0])
df['WindGustDir'] = df['WindGustDir'].fillna(df['WindGustDir'].mode()[0])
df['WindDir3pm'] = df['WindDir3pm'].fillna(df['WindDir3pm'].mode()[0])

#Drop variables with a lot of missing data, or no correlation
df = df.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','Date', 'WindGustDir', 'WindDir3pm', 'WindDir9am'],axis=1)
df.shape

#See unique values and convert them to int using pd.getDummies()
#categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']
#for col in categorical_columns:
#    print(np.unique(df[col]))
# transform the categorical columns
#df = pd.get_dummies(df, columns=categorical_columns)
#df.iloc[4:9]

# simply change yes/no to 1/0 for RainToday and RainTomorrow
df['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
df['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)


#Finding different data types
df.info()

#standardize variables
standa = preprocessing.MinMaxScaler()
standa.fit(df)
datafinal = pd.DataFrame(standa.transform(df), index=df.index, columns=df.columns)
datafinal.head()


#Splitting up dataset
Y = df['RainTomorrow']
X = df.drop(columns=['RainTomorrow'])

# set aside 15% of  data for test
X_train, X_test, y_train, y_test = train_test_split(X, Y,
    test_size=0.30, shuffle = True, random_state = 8)


# Use the same function above for the validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
    test_size=0.15, random_state= 8)

print(f"No. of training examples: {X_train.shape[0]}")
print(f"No. of training examples: {y_train.shape[0]}")

print(f"No. of testing examples: {X_test.shape[0]}")
print(f"No. of testing examples: {y_test.shape[0]}")

print(f"No. of validation examples: {X_val.shape[0]}")
print(f"No. of validation examples: {y_val.shape[0]}")

#Looking into the correlation matrix of all variabels
df.corr()["RainTomorrow"]
train_corr = df.drop(columns=[]).corr(method='pearson')
plt.figure(figsize=(18, 12))
sns.set(font_scale=1.4)
sns.heatmap(train_corr, 
            annot=True, 
            linecolor='white', 
            linewidth=0.5, 
            cmap='magma');




#Problem 1 - Logistic Regression on validation set
sc = StandardScaler()
sc.fit(X_train)
X_val_std = sc.transform(X_val)
X_test_std = sc.transform(X_test)

# We defining the model
logreg = LogisticRegression(C=10, random_state=1, solver='lbfgs', multi_class='ovr')

# We train the model
logreg.fit(X_val_std, y_val)

# We predict target values
Y_predict1 = logreg.predict(X_val_std)

# Test score
score_logreg = logreg.score(X_test_std, y_test)
print(score_logreg) #0.8421330033457078



  


#Probelem 2 - see if you can improve your hyperparameters to improve the performance
logreg = LogisticRegression(C=1000, random_state=100, solver='lbfgs', multi_class='ovr', dual=False, penalty='l2')
logreg.fit(X_val_std, y_val)
Y_predict2 = logreg.predict(X_val_std)
score_logreg2 = logreg.score(X_test_std, y_test)
print(score_logreg2) #0.8422017507676796


logreg = LogisticRegression(C=0.0001, random_state=100, solver='lbfgs', multi_class='ovr')
logreg.fit(X_val_std, y_val)
Y_predict3 = logreg.predict(X_val_std)
score_logreg3 = logreg.score(X_test_std, y_test)
print(score_logreg3) #0.7910307530134286


logreg = LogisticRegression(C=1000, random_state=1, solver='saga', multi_class='ovr', class_weight = 'balanced')
logreg.fit(X_val_std, y_val)
Y_predict4 = logreg.predict(X_val_std)
score_logreg4 = logreg.score(X_test_std, y_test)
print(score_logreg4) #0.7821165039644347





#Problem 3 - KNN on validation set
class KNeighborsClassifiers:
    def __init__(self, k=3):

        self.k = k

    def fit(self, X, y):

        assert len(X) == len(y)
        self.X_train = X
        self.y_train = y

    def distance(self, X1, X2):

        X1, X2 = np.array(X1), np.array(X2)
        distance = 0
        for i in range(len(X1) - 1):
            distance += (X1[i] - X2[i]) ** 2
        return np.sqrt(distance)

    def predict(self, X_test):

        sorted_output = []
        for i in range(len(X_test)):
            distances = []
            neighbors = []
            for j in range(len(self.X_train)):
                dist = self.distance(self.X_train[j], X_test[i])
                distances.append([dist, j])
            distances.sort()
            distances = distances[0:self.k]
            for _, j in distances:
                neighbors.append(self.y_train[j])
            ans = max(neighbors)
            sorted_output.append(ans)

        return sorted_output

    def score(self, X_test, y_test):

        predictions = self.predict(X_test)
        return (predictions == y_test).sum() / len(y_test)

#MODEL 1
# We instantiate my model which uses only numpy.
knn=KNeighborsClassifier(n_neighbors=1)
# We fit the model to the validation data.
knn.fit(X_val,y_val)
# We run some predictions using the test sample data.
predictions=knn.predict(X_test)
# We score the prediction accuracy.
knn.score(X_test,y_test) #0.7826435675328842


#MODEL 2
knn=KNeighborsClassifier(n_neighbors=4)
# We fit the model to the validation data.
knn.fit(X_val,y_val)
# We run some predictions using the test sample data.
predictions2=knn.predict(X_test)
# We score the prediction accuracy.
knn.score(X_test,y_test) #0.8290939089784133


#MODEL 3
knn=KNeighborsClassifier(n_neighbors=10)
# We fit the model to the validation data.
knn.fit(X_val,y_val)
# We run some predictions using the test sample data.
predictions3=knn.predict(X_test)
# We score the prediction accuracy.
knn.score(X_test,y_test) #0.8390164535496586


#MODEL 4
knn=KNeighborsClassifier(n_neighbors=8)
# We fit the model to the validation data.
knn.fit(X_val,y_val)
# We run some predictions using the test sample data.
predictions4=knn.predict(X_test)
# We score the prediction accuracy.
knn.score(X_test,y_test) #0.8381456528713507

knn_score = accuracy_score(y_test,predictions3)
print('Accuracy :',knn_score)




#Problem 4 - Dummy Classifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_val, y_val)
DummyClassifier(strategy='most_frequent')
dummy_clf.predict(X_test)
baseline = dummy_clf.score(X_test, y_test) #0.7832622943306292

dummy_clf2 = DummyClassifier(strategy="stratified")
dummy_clf2.fit(X_val, y_val)
DummyClassifier(strategy='stratified')
dummy_clf2.predict(X_test)
dummy_clf2.score(X_test, y_test) #0.6528942664650076





#Problem 5 - Compare models
testScores = pd.Series([score_logreg2, knn_score, baseline], index=['Logistic', 'KNN', 'Baseline'])
print(testScores)



















