import numpy as np
from sklearn import linear_model

X = [[1, 1, 1],
     [1, 1, 1],
     [0, 0, 0]]
Y = [1, 1, 0]
lr = linear_model.LogisticRegression(C=1, penalty='l2', tol=0.01)
lr.fit(X, Y)

print(lr.coef_)

testData = [[1, 1, 1],
            [0, 0, 0]]
testY = lr.predict(testData)
print(testY)
