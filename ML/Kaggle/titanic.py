from sklearn import linear_model
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# data pre-processing
data = pd.read_csv("C:\\Users\\cjd\\Desktop\\titanic\\train.csv")
trainY = data['Survived']
trainX = data.copy()
del trainX['PassengerId'], trainX['Name'], trainX['Survived'], trainX['Ticket'], trainX['Embarked'], trainX['Cabin']
trainX['Sex'] = (trainX['Sex'] == 'male')
trainX[trainX['Age'].isnull()] = trainX['Age'].mean()

# model selection(Logistic Regression)
lr = linear_model.LogisticRegression()
lr.fit(trainX, trainY)
# print(lr.coef_)

# predict
data = pd.read_csv("C:\\Users\\cjd\\Desktop\\titanic\\test.csv")
testX = data.copy()
del testX['PassengerId'], testX['Name'], testX['Ticket'], testX['Cabin'], testX['Embarked']
testX['Sex'] = (testX['Sex'] == 'male')
testX[testX['Age'].isnull()] = testX['Age'].mean()
testX[testX['Fare'].isnull()] = testX['Fare'].mean()
# print(testX.isnull().sum())
Y = lr.predict(testX)
result = pd.DataFrame({'Survived': Y}, columns={'Survived', 'PassengerId'})
result['PassengerId'] = data['PassengerId']
result.to_csv("C:\\Users\\cjd\\Desktop\\titanic\\output.csv")
# print(result)








