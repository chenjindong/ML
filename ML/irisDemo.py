import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics

iris = datasets.load_iris()
trainX, testX, trainy, testy = cross_validation.train_test_split(iris.data, iris.target, test_size=0.4,)

# data preprocessing(normalization)
scaler = preprocessing.StandardScaler().fit(trainX)
testX_transformed = scaler.transform(testX)

# model selection(SVC)
clf = svm.SVC(kernel='linear')

# model evaluation
scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=4)
print(scores)

predicts = cross_validation.cross_val_predict(clf, iris.data, iris.target, cv=5)
print(metrics.accuracy_score(iris.target, predicts))

kf = cross_validation.KFold(10, 3, shuffle=True, random_state=0)
for train, test in kf:
    print(train, test)



