import pandas as pd
from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble
from sklearn import cross_validation
from sklearn import preprocessing

data = pd.read_csv("C:\\Users\\cjd\\Desktop\\titanic\\train.csv")
trainX = data.drop(['Name', 'Ticket', 'Cabin', 'Survived', 'PassengerId'], axis=1)
trainY = data['Survived'].copy()  # 如果不加copy，则是引用
# 数据预处理
# 空值填充
trainX['Age'] = trainX['Age'].fillna(-1)
trainX['Embarked'] = trainX['Embarked'].fillna('S')
# dummy化非数值型变量
embarked = pd.get_dummies(trainX['Embarked'])
trainX = trainX.drop(['Embarked'], axis=1).join(embarked)  # drop a column and add columns to dataframe
sex = pd.get_dummies(trainX['Sex'])
trainX = trainX.drop(['Sex'], axis=1).join(sex)
# 数据标准化
scaler = preprocessing.StandardScaler().fit(trainX)
trainX = scaler.transform(trainX)

# 模型选择
# logistic regression
lr = linear_model.LogisticRegression()
lr.fit(trainX, trainY)
scores = cross_validation.cross_val_score(lr, trainX, trainY, cv=5)
print(scores.mean())
# SVM
svc = svm.SVC()
svc.fit(trainX, trainY)
scores = cross_validation.cross_val_score(svc, trainX, trainY, cv=5)
print(scores.mean())
# random forest
rf = ensemble.RandomForestClassifier(n_estimators=100)
rf.fit(trainX, trainY)
scores = cross_validation.cross_val_score(rf, trainX, trainY, cv=5)
print(scores.mean())

# 预测
data = pd.read_csv("C:\\Users\\cjd\\Desktop\\titanic\\test.csv")
testX = data.drop(['Name', 'Cabin', 'Ticket', 'PassengerId'], axis=1)
testX['Age'] = testX['Age'].fillna(-1)
testX['Fare'] = testX['Fare'].fillna(testX['Fare'].mean())
embarked = pd.get_dummies(testX['Embarked'])
testX = testX.drop(['Embarked'], axis=1).join(embarked)  # drop a column and add columns to dataframe
sex = pd.get_dummies(testX['Sex'])
testX = testX.drop(['Sex'], axis=1).join(sex)
testX = scaler.transform(testX)

testY = rf.predict(testX)
result = pd.DataFrame({'PassengerId': data['PassengerId'], 'Survived': testY})
result.to_csv("C:\\Users\\cjd\\Desktop\\titanic\\output.csv")
