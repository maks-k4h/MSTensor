import numpy as np

from MSTensor_mcep import *

a = Tensor(np.array([[2, 3]]), True)
b = Tensor(np.array([5, 1]), True)

c = (a @ b).sum()
c.back_prop()

print('Result: ', c)
print('a.grad:\n', a.grad)
print('b.grad:\n', b.grad)
