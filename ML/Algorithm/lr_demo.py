from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


dataset = load_breast_cancer()
x = dataset['data']
y = dataset['target']
train_samples = int(len(y)*0.8)  # 80% training data; 20% test data

x_train, y_train = x[:train_samples], y[:train_samples]
x_test, y_test = x[train_samples:], y[train_samples:]

lr = LogisticRegression()

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
