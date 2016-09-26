import numpy as np
from sklearn import linear_model

X = [[1, 1, 1],
     [1, 1, 1],
     [0, 0, 0]]
Y = [1, 1, 0]
lr = linear_model.LogisticRegression()
lr.fit(X, Y)

print(lr.coef_)

testData = [[1, 1, 1],
            [0, 0, 0]]
testY = lr.predict(testData)
print(testY)
