from sklearn import linear_model
clf = linear_model.LinearRegression()
x = [[0, 0], [1, 1], [2, 2]]
y = [0, 1, 2]
clf.fit(x, y)
print(clf.coef_)