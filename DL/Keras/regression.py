import numpy as np
np.random.seed(1337)
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense      # 全连接层

x = np.linspace(-1, 1, 200)
np.random.shuffle(x)
y = 0.5*x + 2 + np.random.normal(0, 0.05, (200,))

#plt.scatter(x, y)
#plt.show()

x_train, y_train = x[:160], y[:160]
x_test, y_test = x[160:], y[160:]

model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))
model.compile(loss='mse', optimizer='sgd')

# train
for step in range(300):
    cost = model.train_on_batch(x_train, y_train)
    if step%100 == 0:
        print('train cost ', cost)

# test
cost = model.evaluate(x_test, y_test, batch_size=40)
print('test cost ', cost)
W, b = model.layers[0].get_weights()
print('weights=', W, '\nbiases=', b)
y_predict = model.predict(x_test)




















