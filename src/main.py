from src.mnist import init
from src.mnist import load
from src.network import Network
import numpy as np
import src.mnist_loader as mnist_loader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# def test():
#     train_X = np.array([[0, 0],
#                         [0, 1],
#                         [1, 0],
#                         [1, 1]])
#
#     train_Y = np.array([0,
#                         0,
#                         0,
#                         1])
#
#     architecture = [{'size': 2, 'activation': 'sigmoid'},
#                     {'size': 2, 'activation': 'sigmoid'}]
#
#     net = Network(architecture, 'mse', seed=1)
#
#     net.fit(train_X, train_Y, 500, 2, 3.0, train_X, train_Y)


def main():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    architecture = [{'size': 784, 'activation': 'sigmoid'},
                    {'size': 30, 'activation': 'sigmoid'},
                    {'size': 10, 'activation': 'sigmoid'}]

    net = Network(architecture, 'mse', seed=1)

    train_X = np.array([pair[0] for pair in training_data]).reshape(50000, 784)
    train_Y = np.array([pair[1] for pair in training_data]).reshape(50000, 10)
    test_X = np.array([pair[0] for pair in test_data]).reshape(10000, 784)
    test_y = np.array([pair[1] for pair in test_data]).reshape(10000)

    net.fit(train_X, train_Y, 30, 30, 80.0, test_X, test_y)


if __name__ == '__main__':
    main()
