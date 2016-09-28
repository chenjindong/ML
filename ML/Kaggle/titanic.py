from sklearn import linear_model
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import cross_validation
from sklearn import svm


# data pre-processing
data = pd.read_csv("C:\\Users\\cjd\\Desktop\\titanic\\train.csv")
trainY = data['Survived']
trainX = data.copy()
del trainX['PassengerId'], trainX['Name'], trainX['Survived'], trainX['Ticket'], trainX['Cabin']
trainX['Sex'] = (trainX['Sex'] == 'male')
trainX = trainX.fillna(trainX.mean())  # 填充空值的方法
embarked = pd.get_dummies(trainX['Embarked'])  # dummy variable
del trainX['Embarked']
for i in np.arange(3):
    trainX[embarked.columns.values[i]] = embarked[embarked.columns.values[i]]

# model selection
# Logistic Regression
lr = linear_model.LogisticRegression()
lr.fit(trainX, trainY)
# scores = cross_validation.cross_val_score(lr, trainX, trainY, cv=5)
# print(scores)

# SVM
svm1 = svm.SVC()
svm1.fit(trainX, trainY)
# scores = cross_validation.cross_val_score(svm1, trainX, trainY, cv=5)
# print(scores)

# predict
data = pd.read_csv("C:\\Users\\cjd\\Desktop\\titanic\\test.csv")
testX = data.copy()
del testX['PassengerId'], testX['Name'], testX['Ticket'], testX['Cabin']
testX['Sex'] = (testX['Sex'] == 'male')
testX = testX.fillna(testX.mean())  # 填充空值的方法
embarked = pd.get_dummies(testX['Embarked'])  # dummy variable
del testX['Embarked']
for i in np.arange(3):
    testX[embarked.columns.values[i]] = embarked[embarked.columns.values[i]]


Y = lr.predict(testX)
result = pd.DataFrame({'Survived': Y}, columns={'PassengerId', 'Survived'})
result['PassengerId'] = data['PassengerId']
result.to_csv("C:\\Users\\cjd\\Desktop\\titanic\\output.csv")
# print(result)


