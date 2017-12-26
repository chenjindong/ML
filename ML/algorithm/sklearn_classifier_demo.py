from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

dataset = load_breast_cancer()
x = dataset['data']
y = dataset['target']
train_samples = int(len(y)*0.8)  # 80% training data; 20% test data

x_train, y_train = x[:train_samples], y[:train_samples]
x_test, y_test = x[train_samples:], y[train_samples:]

def evaluate(y_test, y_pred):
    precision = precision_score(y_true=y_test, y_pred=y_pred)
    recall = recall_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)

    print('accuracy', acc)
    print('precison: ', precision)
    print('recall: ', recall)
    print('f1: ', f1)


def lr_classifier():
    print(lr_classifier.__name__)
    clf = LogisticRegression()
    model = clf.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    evaluate(y_test, y_pred)
lr_classifier()

def svm_classifier():
    print(svm_classifier.__name__)
    clf = SVC()
    model = clf.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    evaluate(y_test, y_pred)
svm_classifier()


def rf_classifier():
    print(rf_classifier.__name__)
    clf = RandomForestClassifier()
    model = clf.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    evaluate(y_test, y_pred)
rf_classifier()

def xgboost_classifier():
    print(xgboost_classifier.__name__)
    clf = XGBClassifier()
    model = clf.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    evaluate(y_test, y_pred)
xgboost_classifier()



