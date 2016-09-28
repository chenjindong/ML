# [ 0.77653631  0.7877095   0.78089888  0.76404494  0.81920904] Logistic Regression
# [ 0.83798883  0.82122905  0.80898876  0.79775281  0.85875706] SVM with RBF
# [ 0.83798883  0.82122905  0.80898876  0.79775281  0.85875706] SVM with linear Kernal
from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn import cross_validation
from sklearn import preprocessing

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
scaler = preprocessing.StandardScaler().fit(trainX)  # 数据归一化
trainX = scaler.transform(trainX)


# model selection
# Logistic Regression
lr = linear_model.LogisticRegression()
lr.fit(trainX, trainY)
scores = cross_validation.cross_val_score(lr, trainX, trainY, cv=5)
print(scores)

# SVM
svc = svm.SVC()
svc.fit(trainX, trainY)
scores = cross_validation.cross_val_score(svc, trainX, trainY, cv=5)
print(scores)

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
testX = scaler.transform(testX)

Y = svc.predict(testX)
result = pd.DataFrame({'Survived': Y}, columns={'PassengerId', 'Survived'})
result['PassengerId'] = data['PassengerId']
result.to_csv("C:\\Users\\cjd\\Desktop\\titanic\\output.csv")
# print(result)
