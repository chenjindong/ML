import os, re, time
import jieba
import cjdpy
import numpy as np
import requests
import json
from collections import Counter
start = time.clock()

"""
	dataset: vocabulary size, concept size
	emotion_analysis: 5000, 738
	news_title: 5000, 750
	sentiment_classification: 15000, 1280
	topic_classification:30000, 6000
"""
task = 'news_title'
vocab_size = {'emotion_analysis': 5000, 'news_title': 5000,
              'sentiment_classification': 14000,'topic_classification': 30000}
con_size = {'emotion_analysis': 750, 'news_title': 800,
              'sentiment_classification': 1300,'topic_classification': 6000}

train_file = 'data/%s/train.txt' % task
test_file = 'data/%s/test.txt' % task
vocab_file = 'data/vocabulary/%s.txt' % task
con_file = 'data/vocabulary/%s.con.txt' % task

vocab_size = vocab_size[task]
con_size = con_size[task]
maxSeqLen = 20
maxConLen = 5

def wordCount():
	data = cjdpy.load_csv(train_file)
	X = [list(jieba.cut(item[0])) for item in data]
	vocab_list = [word for text in X for word in text]
	counter = Counter(vocab_list)
	counter = cjdpy.sort_list_by_freq(counter)
	cjdpy.save_csv(counter, vocab_file)

def conCount():
	data = cjdpy.load_csv(train_file)
	vocab_list = [item[3].split('|||') for item in data if len(item)==4]
	vocab_list = sum(vocab_list, [])
	counter = Counter(vocab_list)
	counter = cjdpy.sort_list_by_freq(counter)
	cjdpy.save_csv(counter, con_file)

def make_vocab():
	global id2w, w2id
	vocab_list = cjdpy.load_csv(vocab_file)
	id2w = [vocab_list[i][0] for i in range(vocab_size)]
	w2id = {word: i for i, word in enumerate(id2w)}
	# tip: set the vocabulary size according to different dataset
	id2w = ['<PAD>', '<UNK>'] + [x[0] for x in vocab_list[:vocab_size]]
	w2id = {y: x for x, y in enumerate(id2w)}
	return id2w, w2id

def pre_trained_embedding():
    # embedding_matrix = np.random.normal(size=(vocab_size+2, 50))
    embedding_matrix = np.zeros((vocab_size + 2, 50))
    data = cjdpy.load_csv('data/vectors_word.txt', ' ')
    w2v = {item[0]: np.array(list(map(lambda x: float(x), item[1:]))) for item in data[1:]}
    for i in range(vocab_size + 2):
        emb = w2v.get(id2w[i])
        if emb is not None: embedding_matrix[i, :] = emb
        else: print(id2w[i])
    return embedding_matrix

def make_con_vocab():
	global id2con, con2id
	vocab_list = cjdpy.load_csv(con_file)
	cons = [vocab_list[i][0] for i in range(con_size)]
	cons.remove('PAD')
	id2con = ['<PAD>', '<UNK>'] + cons
	con2id = {y: x for x, y in enumerate(id2con)}
	return id2con, con2id

def make_label(file):
	global id2label, label2id
	data = cjdpy.load_csv(file)
	label = [item[1] for item in data if len(item) == 4]
	id2label = list(set(label))
	label2id = {y: x for x, y in enumerate(id2label)}
	return id2label, label2id

def Tokens2Intlist(word2id, tokens, maxSeqLen):

	ret = np.zeros(maxSeqLen)
	tokens = tokens[:maxSeqLen]
	for i, t in enumerate(tokens):
		ret[maxSeqLen - len(tokens) + i] = word2id.get(t, 1)
	return ret

def load_data(file):
	data = cjdpy.load_csv(file)
	X, y = [], []
	for text, label, ent, con in data:
		try:
			words = list(jieba.cut(text))
			cons = con.split('|||')
			X.append(Tokens2Intlist(w2id, words, maxSeqLen).tolist() + Tokens2Intlist(con2id, cons, maxConLen).tolist())  # with prior concept
			y.append(label2id[label])
		except Exception as e:
			import traceback
			traceback.print_exc()

	# sample imbalance problem
	# class_weight = Counter([item[1] for item in data])
	# class_weight = {label2id[key]:val/len(y) for key, val in class_weight.items()}
	return X, y

def entity_linking(path):
	data = cjdpy.load_csv(path)
	res = []
	fout = open('out1.txt', 'w', encoding='utf-8')
	for i, item in enumerate(data):
		if i <= 83429: continue
		if len(item) == 2:
			text, label = item[0], item[1]
		else: continue
		if i % 500 == 0: print('processing %d...' %i)
		url = 'http://shuyantech.com/api/entitylinking/cutsegment?q=%s&apikey=%s' % (text,'ljqljqljq')
		try: response = json.loads(requests.get(url).text)
		except: print('entity linking fail: ', text)
		ents = [item[1] for item in response.get('entities', [])]
		if len(ents) == 0: ents=['PAD']
		fout.write(text + '\t' + label + '\t' + '|||'.join(ents) + '\n')
		fout.flush()
		# res.append([text, label, '|||'.join(ents)])
	# cjdpy.save_csv(res, 'st_ent.txt')

# entity_linking()

def get_concept():
	data = cjdpy.load_csv('st_ent.txt')
	res = []
	for i, (text, lable, ents) in enumerate(data):
		if i % 500 == 0: print('processing %d...' %i)
		cons = []
		for item in ents.split('|||'):
			url = 'http://shuyantech.com/api/cnprobase/concept?q=%s&apikey=%s' % (item, 'ljqljqljq')
			response = json.loads(requests.get(url).text)
			cons += [item[0] for item in response.get('ret', [])]
		if len(cons) == 0: cons=['PAD']
		res.append([text, lable, ents, '|||'.join(cons)])
	cjdpy.save_csv(res, 'text.con.txt')
# get_concept()

if __name__ == '__main__':
	# entity_linking('data/topic_classification/text.txt')
	# get_concept()

    # output the short text
    # make_vocab()
    # make_con_vocab()
    # make_label(train_file)
    # X, y = load_data(test_file)
    # line = 4
    # for i in range(len(X[line])):
		# if i>maxSeqLen: print(id2con[int(X[line][i])])
		# else: print(id2w[int(X[line][i])])

    # embedding test
    make_vocab()
    embeddings = pre_trained_embedding()
    for i in range(5):
        print(embeddings[i, :])

print(time.clock()-start)
