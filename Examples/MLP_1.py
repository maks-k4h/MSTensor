import random
import numpy as np
from MSTensor import *

# ====================================
# Simple Multilayer Perceptron Example
# ====================================


# We'll learn simple binary function using MLP
def f(x):
    return x[0] | x[1]


data = [
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1]
]

random.shuffle(data)

labels = [f(x) for x in data]

# training data is not enough to certainly state the
# function's closed form, since to do so we need all
# 2^4 = 16 examples. We divide data 75/25
# training/testing respectively
data_train, data_test = data[:12], data[12:]
labels_train, labels_test = labels[:12], labels[12:]


# NLP structure:
# I -> L -> O -> E, where
# I — input layer
# L — hidden layer with ReLU activation function
# O — output layer that is a sigmoid representing the probability of getting 1
# E — loss function — cross-entropy, since maximum likelihood estimation
#     is more convenient here.

# I — not really a layer, it will be given by matrix X of input labels
# L(X, W1, B1) = ReLU(X @ W1 + B1)
# O(L, W2, B2) = sigmoid(L @ W2 + B2)
# E(O, Y) = -sum(Y*log(O) + (1-Y)*log(1-O)) / n, where
#   Y is a matrix of labels,
#   E is just mean cross entropy -(y*log(o) + (1-y)*log(1-o)),
#   n is the number of input examples.

# let us introduce one hyperparameter of the model:
J = 4  # the numer of neurons in first (and last) hidden layer

# model initialization
W1 = Tensor(np.random.rand(4, J), True)
W2 = Tensor(np.random.rand(J), True)
B1 = Tensor(np.random.rand(1, J), True)
B2 = Tensor(np.random.rand(1), True)
N = Tensor(np.array([len(data_train)]))
X = Tensor(np.array(data_train))
Y = Tensor(np.array(labels_train))
one = Tensor(np.ones(1))

# Training the model
speed = .9
for i in range(1000):
    L = ReLU(X @ W1 + B1)
    O = sigmoid(L @ W2 + B2)
    E = -(Y*log(O) + (one - Y)*log(one - O)).sum() / N
    E.back_prop()
    if not (i + 1) % 100:
        print(i+1, '\tCross-Entropy: ', round(E.value[0][0], 5))

    # simplest gradient descent
    W1.value -= W1.grad * speed
    W2.value -= W2.grad * speed
    B1.value -= B1.grad * speed
    B2.value -= B2.grad * speed


# Time for predicting results on unseen inputs
print('\n\nLabel\tPredicted*')
for x, y in zip(data_test, labels_test):
    L = ReLU(Tensor(np.array(x).reshape(1, 4)) @ W1 + B1)
    O = sigmoid(L @ W2 + B2)
    print(y, round(O.value[0][0], 3), sep='\t\t')
print('\n* is the conditional probability of P(y=1|x).')

