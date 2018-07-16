import cjdpy
import jieba
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.metrics import classification_report_imbalanced

"""
vocabulary size
    emotion_analysis: 5000
    news_title: 5000
    sentiment_classification: 15000
    topic_classification:30000
"""
task = 'topic_classification'
train_file = 'data/%s/train.txt' % task
test_file = 'data/%s/test.txt' % task
vocab_file = 'data/vocabulary/%s.txt' % task
vocab_size = {'emotion_analysis': 5000, 'news_title': 5000,
              'sentiment_classification': 14000,'topic_classification': 30000}

vocab_size = vocab_size[task]

def word_count():
    data = cjdpy.load_csv(train_file)
    X, y = [], []
    for item in data:
        X.append(list(jieba.cut(item[0])))
        # y.append(item[1])
    vocab_list = [word for text in X for word in text]
    counter = Counter(vocab_list)
    counter = cjdpy.sort_list_by_freq(counter)
    cjdpy.save_csv(counter, vocab_file)
    assert False

def bag_of_words(X):
	 X_vec = np.zeros((len(X), vocab_size))
	 for i in range(len(X)):
		 for token in X[i]:
			 if w2id.get(token):
				 X_vec[i, w2id[token]] += 1
	 return X_vec

def load_data(file):
    data = cjdpy.load_csv(file)
    X, y = [], []
    for item in data:
        X.append(list(jieba.cut(item[0])))
        y.append(item[1])
    X = bag_of_words(X)
    return X, y


vocab_list = cjdpy.load_csv(vocab_file)
id2w = [vocab_list[i][0] for i in range(vocab_size)]
w2id = {word: i for i, word in enumerate(id2w)}

X_train, y_train = load_data(train_file)
X_test, y_test = load_data(test_file)
print('training samples: ', len(X_train))
print('test samples: ', len(X_test))

clf = LinearSVC()
# clf = LogisticRegression()
clf.fit(X_train, y_train)


def evaluate(y_true, y_pred):
	 print(classification_report_imbalanced(y_true, y_pred))
	 print('accuracy: ', accuracy_score(y_true, y_pred))

y_pred = clf.predict(X_test)
for i in range(20):
    print(i, y_test[i], y_pred[i])

evaluate(y_test, y_pred)

