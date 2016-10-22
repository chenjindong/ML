# SGD [ 0.87293278  0.84755444  0.86927015  0.87959986  0.86969986]
# rf [ 0.95407496  0.95370701  0.95118466  0.9528403   0.95640781]
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import svm
from sklearn import ensemble

data = pd.read_csv("C:\\Users\\cjd\\Desktop\\data\\Digit_Recognizer\\train.csv")
data = data.fillna(0)
cols = data.columns[data.sum() < 4000].tolist()
print(len(cols))


train = data.drop(cols, axis=1)
train = train.drop(['label'], axis=1)


lr = linear_model.LogisticRegression()
slr = linear_model.SGDClassifier()
svc = svm.SVC()
rf = ensemble.RandomForestClassifier(n_estimators=50)
gbdt = ensemble.GradientBoostingClassifier()
# lr.fit(train, data['label'])
gbdt.fit(train, data['label'])

scores = cross_validation.cross_val_score(gbdt, train, data['label'], cv=2)
print(scores)

data = pd.read_csv("C:\\Users\\cjd\\Desktop\\data\\Digit_Recognizer\\test.csv")
data = data.fillna(0)
test = data.drop(cols, axis=1)
predict = gbdt.predict(test)
result = pd.DataFrame({'ImageId': np.arange(len(predict))+1, 'Label': predict})
result = result.to_csv("C:\\Users\\cjd\\Desktop\\data\\Digit_Recognizer\\result.csv")






