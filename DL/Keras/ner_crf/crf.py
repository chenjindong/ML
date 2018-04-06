import pandas as pd
import numpy as np
import nltk
import time
from sklearn_crfsuite import CRF
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn_crfsuite.metrics import flat_classification_report
'''
ref:
https://www.depends-on-the-definition.com/named-entity-recognition-conditional-random-fields-python/

8个类

'''
start = time.time()
# train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
# test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
# # print(train_sents[0])
# # print(len(train_sents))
# y = set()
# for sent in train_sents:
#     for item in sent:
#         y.add(item[2])
# print(y)
# assert False

data = pd.read_csv("data/ner_dataset.csv", encoding="latin1")

data = data.fillna(method="ffill")  # fill with the field of last row

# print(data.head(50))

words = list(set(data["Word"].values))

# print(len(words))

class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(data)
sent = getter.get_next()
sentences = getter.sentences  # 每个元素是一个句子的word集合， e.g., [('Thousands', 'NNS', 'O'), ('of', 'IN', 'O')]
print(sentences[:1])
# assert False


def word2features(sent, i):
    '''
    给句子中的每个词 找 特征
    '''
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]  # label sequence, e.g., ['O', 'O', 'B-geo', 'O', 'O', 'O', 'B-geo', 'O']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X[:1][0])
# print(y[:1][0])


# assert False
crf = CRF(algorithm='lbfgs', min_freq=1,
          c1=0.1,
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=False)

crf.fit(X_train, y_train)
pred = crf.predict(X_test)

report = flat_classification_report(y_pred=pred, y_true=y_test)
print(report)

# res = crf.predict(X_test[:4])
# print(res)
# res = res[0]
# for i in range(len(res)):
#     print(X_test[0][i], res[i])

print(time.time()-start)
