import numpy as np
import sklearn.utils as sp
from numpy import linalg as LA
from utils import Log
import matplotlib.pyplot as plt


class Layer(object):
    def __init__(self, input_size, output_size, activation):
        """activation - sigmoid"""
        self.b = np.random.randn(output_size, 1)
        self.W = np.random.randn(output_size, input_size)

        # self.b = np.zeros(shape=(output_size, 1), dtype=np.float64) / 2
        # self.W = np.ones(shape=(output_size, input_size), dtype=np.float64) / 2

        if activation == 'sigmoid':
            self.activation = sigmoid
            self.d_activation = d_sigmoid


class Network(object):
    def __init__(self, architecture, loss, seed=None):
        """
        architecture - [{size: 10, activation: sigmoid}, {size: 12, activation: relu}]
        loss - mse
        """
        if seed:
            np.random.seed(seed)

        self.layers = [Layer(architecture[i]['size'], architecture[i + 1]['size'], architecture[i + 1]['activation'])
                       for i in range(len(architecture) - 1)]

        self.num_layers = len(architecture)

        if loss == 'mse':
            self.loss = mse
            self.d_loss = d_mse

        self.log = Log('network.log')

    def predict(self, X):
        for layer in self.layers:
            X = layer.activation(np.dot(layer.W, X) + layer.b.dot(np.ones((1, X.shape[1]))))
        return X

    def fit(self, train_X, train_Y, epochs, batch_size, learning_rate, test_X=None, test_y=None):
        for j in range(epochs):
            #train_data = sp.shuffle(train_X, train_Y)
            train_data = (train_X, train_Y)

            mini_batches_X = [train_data[0][k: k + batch_size] for k in range(0, len(train_X), batch_size)]
            mini_batches_Y = [train_data[1][k: k + batch_size] for k in range(0, len(train_Y), batch_size)]

            #self.log.print([self.layers[0].W, self.layers[0].b])

            # for x in mini_batches_X:
            #     self.paint(x)

            for mini_batch_X, mini_batch_Y in zip(mini_batches_X, mini_batches_Y):
                self.update_mini_batch(mini_batch_X.transpose(), mini_batch_Y.transpose(), learning_rate)

                # self.log.print([self.layers[0].W, self.layers[0].b])
                # pass

            if test_X is not None and test_y is not None:
                #self.log.print(['end of fit\n', self.layers[0].W, self.layers[0].b])

                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_X.transpose(), test_y.transpose()), len(test_X)))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, X, Y, eta):
        d_bs, d_ws = self.backpropagation(X, Y)

        # if self.log.counter > 0:
        #     self.log.print(['after backpropagation\n', d_bs, d_ws])

        for l in range(len(self.layers)):
            self.layers[l].W -= eta / (Y.shape[1]) * d_ws[l]
            self.layers[l].b -= eta / (Y.shape[1]) * d_bs[l]

    def backpropagation(self, X, Y):
        d_bs = [np.zeros(layer.b.shape) for layer in self.layers]
        d_Ws = [np.zeros(layer.W.shape) for layer in self.layers]

        # feedforward

        activations = [X]
        zs = []

        for layer in self.layers:
            zs.append(np.dot(layer.W, activations[-1]) + layer.b.dot(np.ones((1, X.shape[1]))))
            activations.append(layer.activation(zs[-1]))

        # if self.log.counter > 0:
        #     for z in zs:
        #         self.log.print(['after forward z\n', z])
        #     for a in activations:
        #         self.log.print(['after forward a\n', a])

        # backward pass

        delta = self.d_loss(activations[-1], Y) * self.layers[-1].d_activation(zs[-1])

        # if self.log.counter > 0:
        #     self.log.print(['delta\n', delta])

        d_bs[-1] = delta.dot(np.ones(shape=(delta.shape[1], 1)))
        d_Ws[-1] = delta.dot(activations[-2].transpose())

        # if self.log.counter > 0:
        #     self.log.print(['delta_b, delta_w\n', d_bs, d_Ws])

        for l in range(2, self.num_layers):
            delta = self.layers[-l+1].W.transpose().dot(delta) * self.layers[-l].d_activation(zs[-l])
            d_bs[-l] = delta.dot(np.ones(shape=(delta.shape[1], 1)))
            d_Ws[-l] = delta.dot(activations[-l-1].transpose())

            # if self.log.counter > 0:
            #     self.log.print(['delta_b, delta_w\n', d_bs, d_Ws])

        return d_bs, d_Ws

    def evaluate(self, X, y):
        A = np.argmax(self.predict(X), axis=0)

        return sum(A == y)

    def vectorized(self, y):
        Y = np.zeros(shape=(len(y), self.layers[-1].W.shape[0]))
        for i in range(len(y)):
            Y[i][y[i]] = 1
        return Y

    def paint(self, arr):
        img = arr.reshape(28, 28, 1)
        img2 = np.zeros(shape=(28, 28, 4))
        for i in range(28):
            for j in range(28):
                for k in range(4):
                    img2[i, j, k] = img[i, j, 0]
        plt.imshow(img2)
        plt.show()


def mse(A, Y):
    error = Y - A
    return 0.5 / Y.shape[0] * LA.norm(error, 'fro') ** 2


def d_mse(A, Y):
    return (A - Y) / Y.shape[0]


def sigmoid(Z):
    return 1.0/(1.0 + np.exp(-Z))


def d_sigmoid(Z):
    sig = sigmoid(Z)
    return sig * (1 - sig)
