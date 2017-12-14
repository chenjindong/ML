from keras.models import Model
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import RepeatVector, Dense
from keras.layers.wrappers import TimeDistributed

'''
摘要生成
learn from this code:
1. RepeatVector
2. LSTM(unit, return_sequence=True)
3. TimeDistributed
'''
class Seq2seq:
    def __init__(self):
        self.src_txt_length = 200
        self.sum_text_length = 50
        self.vocab_size = 10000

    def model1(self):
        '''
        one-shot model
        '''
        # encoder input model
        inputs = Input(shape=(self.src_txt_length,))  # (None, input_length)
        encoder1 = Embedding(self.vocab_size, 128)(inputs)  # (None, input_length, 128)
        encoder2 = LSTM(100)(encoder1)  # (None, 100)
        # repeatvector: make sure that the output dimension meets our expectation
        encoder3 = RepeatVector(self.sum_text_length)(encoder2)  # (None, sum_text_length, 100)

        # decoder output model
        # param return_sequences: Whether to return the last output in the output sequence
        decoder1 = LSTM(100, return_sequences=True)(encoder3)  # (None, 50, 100)
        # TimeDistributed: This wrapper allows us to apply a layer to every temporal slice of an input
        # 将layer应用到每个输入的时间片
        outputs = TimeDistributed(Dense(self.vocab_size, activation='softmax'))(decoder1)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.summary()

    def model2(self):
        # encoder
        inputs1 = Input(shape=(self.src_txt_length,))
        article1 = Embedding(self.vocab_size, 128)(inputs1)
        article2 = LSTM(128)(article1)

        inputs2 = Input(shape=(self.sum_text_length, ))
        summary1 = Embedding(self.vocab_size, 128)(inputs2)
        summary2 = LSTM(128)(summary1)

        # decoder
        decoder = concatenate([article2, summary2])
        decoder = LSTM(128)(decoder)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder)
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.summary()


if __name__ == '__main__':
    seq2seq = Seq2seq()
    seq2seq.model1()
    seq2seq.model2()
