from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, log_loss
from sklearn import preprocessing
import numpy as np

'''
自己实现逻辑回归LR
learned:
    · 知道了数据归一化的重要性，一开始没归一化，很难收敛归一化之后就特别容易收敛
    · 数据归一化的方法有很多种，但是基本就用scale to unit length，
      参考 https://en.wikipedia.org/wiki/Feature_scaling
    · 涉及矩阵乘法时候要用np.mat，不要用np.array
    · log_loss相当于keras中的binary_entropy
    
'''
eta = 0.001  # learning rate
split_rate = 0.7  # training set and validation set rate


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

class LR:
    def __init__(self):
        # dataset
        nums = int(split_rate*len(y))
        self.x_train, self.y_train = np.mat(x[:nums]), np.mat(y[:nums]).transpose()
        self.x_test, self.y_test = np.mat(x[nums:]), np.mat(y[nums:]).transpose()
        # 数据按列归一化
        self.x_train = preprocessing.normalize(self.x_train, axis=0)
        self.x_test = preprocessing.normalize(self.x_test, axis=0)

        # parameter
        self.w = np.mat(np.random.uniform(low=-0.01, high=0.01, size=(30,1)))

    def cal_acc(self, epoches=None):
        # cal acc and val_acc
        z = self.x_train * self.w
        prob = sigmoid(z)
        y = prob >= 0.5
        acc = accuracy_score(self.y_train, y)
        loss = log_loss(self.y_train, y)


        z = self.x_test * self.w
        prob = sigmoid(z)
        y = prob >= 0.5
        val_acc = accuracy_score(self.y_test, y)
        val_loss = log_loss(self.y_test, y)
        print('epoch: %i ;acc: %f; loss: %f; val_acc: %f; loss: %f' % (epoches, acc, loss, val_acc, val_loss))

    # 梯度下降
    def train_gd(self):
        for i in range(3000):
            hx = sigmoid(self.x_train*self.w)
            error = self.y_train-hx
            self.w = self.w + eta*self.x_train.transpose()*error
            self.cal_acc(i)

    # 随机梯度下降
    def train_sgd(self):
        for i in range(1000):
            for j in range(len(self.x_train)):
                hx = sigmoid(self.x_train[j, :]*self.w)
                error = self.y_train[j] - hx  # loss关于w的导数
                # w = w + eta*（loss关于w的导数）
                self.w = self.w + eta*self.x_train[j, :].reshape((30, 1))*error
            self.cal_acc(i)

if __name__ == '__main__':

    data = load_breast_cancer()
    x, y = data['data'], data['target']

    model = LR()
    # model.train_gd()
    model.train_sgd()



