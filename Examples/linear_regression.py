import numpy as np

from MSTensor import *

# simple linear regression
# suppose relationship y = 3*x1 + 2*x2 - 1

X = Tensor(np.array([
    [1, 1],
    [1, 2],
    [3, 3],
    [5, 2]
]))

Y = Tensor(np.array([
    [4],
    [6],
    [14],
    [18]
]))

two = Tensor(np.array([2]))

def mse(x: Tensor, y: Tensor, w: Tensor, b: Tensor):
    A = x @ w
    P = A + b
    return ((P - y) ** two).sum() / Tensor(np.array(x.value.shape[0]))


W, B = Tensor(np.random.rand(2), True), Tensor(np.random.rand(1), True)

print('Initial MSE: {}'.format(mse(X, Y, W, B)))

for i in range(500):
    e = ((X @ W + B - Y) ** two).sum() / Tensor(np.array(X.value.shape[0]))
    e.back_prop()
    W.value -= W.grad * 0.05
    B.value -= B.grad * 0.05

print('Final MSE: {}'.format(mse(X, Y, W, B)))
print('Final Weights: {}, {}'.format(W, B))


