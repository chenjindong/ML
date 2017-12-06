import numpy as np
from keras.models import Model, load_model
from keras.layers.core import Dense
from keras.engine.topology import Input

'''
keras save model and load model
'''
class ModelSaveLoad:
    def __init__(self, x, y):
        self.train_x, self.train_y = x[:-1000], y[:-1000]
        self.test_x, self.test_y = x[-1000:], y[-1000:]

    def train(self):
        print('train model......')
        input = Input(shape=(1,))
        output = Dense(units=1)(input)
        self.model = Model(inputs=input, outputs=output)
        self.model.compile(optimizer='adam', loss='mse')
        self.model.summary()
        self.model.fit(self.train_x, self.train_y, batch_size=64, epochs=20, verbose=2,
                       validation_data=(self.test_x, self.test_y))

    def save_model(self):
        print('save model......')
        layer = self.model.get_layer(name='dense_1')
        print('weight: ', layer.get_weights())
        self.model.save('model.h5')

    def load_model(self):
        print('load model......')
        model_new = load_model('model.h5')
        layer = model_new.get_layer('dense_1')
        print(layer.get_weights())

if __name__ == '__main__':
    x = np.linspace(-1, 1, 10000)
    y = 2*x + 0.5 + np.random.uniform(-0.1, 0.1)
    model = ModelSaveLoad(x, y)
    model.train()
    model.save_model()
    model.load_model()

