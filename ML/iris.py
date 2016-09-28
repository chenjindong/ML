import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn import preprocessing

iris = datasets.load_iris()
trainX, testX, trainy, testy = cross_validation.train_test_split(iris.data, iris.target, test_size=0.4,)
scaler = preprocessing.StandardScaler().fit(trainX)
testX_transformed = scaler.transform(testX)
print(testX_transformed.mean(axis=0))

#
# clf = svm.SVC(kernel='linear')
# scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5)
# print(scores)
