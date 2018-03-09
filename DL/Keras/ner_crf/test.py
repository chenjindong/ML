

def pad_sequences_test():
    from keras.preprocessing.sequence import pad_sequences
    # keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32',
    #     padding='pre', truncating='pre', value=0.)
    X = [[1,2,3],[4,5],[6],[7,9]]
    res = pad_sequences(X, maxlen=10)
    print(res)

def to_cateogrical_test():
    from keras.utils import to_categorical
    y = [0, 1,3]
    res = to_categorical(y)
    print(res)
to_cateogrical_test()


