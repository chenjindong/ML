from gensim.models import Word2Vec
import time

start = time.clock()

class Word2VecGensim:
    def __init__(self, sentences):
        self.sentences = sentences

    def train(self):
        self.model = Word2Vec(self.sentences, size=128)

    def save_model(self):
        print('save model ......')
        self.model.save('embeddings.h5')

    def load_model(self):
        print('load model ......')
        return Word2Vec.load('embeddings.h5')

if __name__ == '__main__':
    path = r'\\10.141.208.22\data\Chinese_isA\corpus\wikicorpus_seg.txt'
    sentences = []
    # train model
    with open(path, 'r', encoding='utf-8') as fin:
        for i, line in enumerate(fin):
            # if i > 100000:
                # break
            tokens = line.strip().split('\t')
            sentences.append(tokens)
    w2v = Word2VecGensim(sentences)
    w2v.train()
    w2v.save_model()

    # load model
    model = w2v.load_model()
    print(model.most_similar('周杰伦'))


print(time.clock()-start)