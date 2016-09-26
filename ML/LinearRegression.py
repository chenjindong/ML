from sklearn import linear_model
clf = linear_model.LinearRegression()
x = [[0, 0], [1, 1], [2, 2]]
y = [0, 1, 2]
clf.fit(x, y)
print(clf.coef_)

testData = [[3, 3],
            [4, 4],
            [5, 6]]
testY = clf.predict(testData)
print(testY)
