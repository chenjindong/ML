from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

'''
logistic regression
    accuracy 0.929824561404
    positive samples' precision, recall, f1
        precison:  0.987804878049
        recall:  0.920454545455
        f1:  0.952941176471
'''

'''
xgboost
    效果确实好
    accuracy 0.973684210526
    positive samples' precision, recall, f1
        precison:  0.988505747126
        recall:  0.977272727273
        f1:  0.982857142857
'''

dataset = load_breast_cancer()
x = dataset['data']
y = dataset['target']
train_samples = int(len(y)*0.8)  # 80% training data; 20% test data

x_train, y_train = x[:train_samples], y[:train_samples]
x_test, y_test = x[train_samples:], y[train_samples:]

lr = XGBClassifier()

model = lr.fit(x_train, y_train)

y_pred = model.predict(x_test)


precision = precision_score(y_true=y_test, y_pred=y_pred)
recall = recall_score(y_true=y_test, y_pred=y_pred)
f1 = f1_score(y_true=y_test, y_pred=y_pred)
acc = accuracy_score(y_true=y_test, y_pred=y_pred)

print('precison: ', precision)
print('recall: ', recall)
print('f1: ', f1)
print('accuracy', acc)