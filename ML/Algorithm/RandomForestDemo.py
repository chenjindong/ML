from sklearn import  ensemble
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = ensemble.RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, Y)
# print(clf)
print(clf.predict([[0, 0]]))