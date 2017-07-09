# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from collections import OrderedDict
import copy

import chainer
from chainer import links as L
from chainer import functions as F
from chainer.dataset.convert import concat_examples

from nelder_mead import NelderMead
import numpy as np


class MLP(chainer.Chain):

    def __init__(self,):
        super(MLP, self).__init__(
            l1=L.Linear(784, 300),
            l2=L.Linear(300, 10)
        )

    def __call__(self, x, t):
        h = F.relu(self.l1(x))
        h = self.l2(h)

        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)

        return self.loss


def main():
    train_data, test_data = chainer.datasets.get_mnist()

    train_iter = chainer.iterators.SerialIterator(
        train_data, 100, repeat=False, shuffle=True)
    test_iter = chainer.iterators.SerialIterator(
        test_data, 100, repeat=False, shuffle=False)

    def train(x):
        epochs = 50
        stepsizes = [40]
        gamma = 0.1
        lr = x[0]
        momentum = x[1]

        model = MLP()
        optimizer = chainer.optimizers.MomentumSGD(lr=lr, momentum=momentum)
        optimizer.setup(model)

        test_accuracy = 0
        for epoch in range(epochs):
            train_iter.reset()
            accuracy = []

            data_iter = copy.copy(train_iter)
            for batch in data_iter:
                x, t = concat_examples(batch)
                optimizer.update(model, x, t)
                accuracy.append(float(model.accuracy.data))
            train_accuracy = np.mean(accuracy)

            del accuracy[:]
            data_iter = copy.copy(test_iter)
            for batch in data_iter:
                x, t = concat_examples(batch)
                model(x, t)
                accuracy.append(float(model.accuracy.data))

            if (epoch + 1) in stepsizes:
                optimizer.lr *= gamma
            test_accuracy = np.mean(accuracy)
            # print(epoch, train_accuracy, test_accuracy)

        return test_accuracy

    hp = OrderedDict()
    hp["lr"] = (0.001, 0.05)
    hp["momentum"] = (0.7, 0.95)

    nm = NelderMead(train, hp)
    nm.initialize([(0.04, 0.88), (0.01, 0.92), (0.02, 0.9)])
    nm.maximize()

if __name__ == "__main__":
    main()
