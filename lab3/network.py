import numpy as np
import random


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x)*(1.0-sigmoid(x))


class Network(object):
    def __init__(self, sizes):
        '''sizes: the number of neurons in each layer'''
        self.biases = [np.random.randn(n, 1) for n in sizes[1:]]
        self.weights = [np.random.randn(n, m)
                        for n, m in zip(sizes[1:], sizes[:-1])]

    def evaluate(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def gradient_desent(self, data, mini_batch_size, eta, epoches):
        '''stochastic gradient descent
        data: [(x, y)]
        '''
        for _ in range(epoches):
            random.shuffle(data)
            mini_batchs = [data[i:i+mini_batch_size]
                           for i in range(0, len(data), mini_batch_size)]
            for mini_batch in mini_batchs:
                self.backprop(mini_batch, eta)

    def backprop(self, mini_batch, eta):
        b_delta = [np.zeros(b.shape) for b in self.biases]
        w_delta = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # feedforward
            zs = []  # z[l] = w[l]*a[l-1]+b[l]
            activations = [x]  # a[l] = sigmoid(z[l])
            a = x
            for b, w in zip(self.biases, self.weights):
                zs.append(np.dot(w, a)+b)
                a = sigmoid(zs[-1])
                activations.append(a)
            # backward pass
            delta = (activations[-1]-y)*sigmoid_derivative(zs[-1])
            b_delta[-1] += delta
            w_delta[-1] += np.dot(delta, activations[-2].transpose())
            for i in range(len(self.weights)-2, -1, -1):
                delta = np.dot(self.weights[i+1].transpose(),
                               delta)*sigmoid_derivative(zs[i])
                b_delta[i] += delta
                w_delta[i] += np.dot(delta, activations[i].transpose())
        self.biases = [b-eta*bd/len(mini_batch)
                       for b, bd in zip(self.biases, b_delta)]
        self.weights = [w-eta*wd/len(mini_batch)
                        for w, wd in zip(self.weights, w_delta)]


iris_id = {0: np.array([[1.0, 0.0, 0.0]]).transpose(),
           1: np.array([[0.0, 1.0, 0.0]]).transpose(),
           2: np.array([[0.0, 0.0, 1.0]]).transpose()}


def load_data(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [l.split(' ') for l in lines]
        return [(np.array([list(map(float, l[:4]))]).transpose(),
                 iris_id[int(l[4])]) for l in lines]


def main():
    train_file = './Iris-train.txt'
    test_file = './Iris-test.txt'
    train_data = load_data(train_file)
    test_data = load_data(test_file)
    accuracies = []
    for _ in range(10):
        network = Network([4, 10, 3])
        network.gradient_desent(train_data, 10, 0.03, 1000)
        cnt = 0
        for x, y in test_data:
            res = network.evaluate(x)
            if np.argmax(y) == np.argmax(res):
                cnt += 1
        acc = cnt/len(test_data)
        print('accuracy:', acc)
        accuracies.append(acc)
    accuracies = np.array(accuracies)
    print('average: {}, standard deviation: {}'.format(
        accuracies.mean(), np.std(accuracies)))


if __name__ == '__main__':
    main()
