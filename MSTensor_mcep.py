import numpy as np
import math

# ========================================================
# MSTensor mcep (more computationally efficient prototype)
# was created since the original version is too slow.
# ========================================================


# Propagators
# ========================================================
# Describe the idea of propagators and main principles
# ========================================================

def leaf_accumulator(tensor, grad):
    """leaf_propagate"""
    if tensor.c_g_:
        if not tensor.is_leaf:
            raise RuntimeError('Not a leaf has the default propagator')
        if tensor.grad is None:
            tensor.grad = grad  # initializing new gradient
        else:
            tensor.grad += grad     # cumulating the gradient


def add_propagate(tensor, grad):
    """add_propagate"""
    if not tensor.c_g_:
        return

    for child in tensor.children_:
        if child is None:
            continue
        """
        Notes:
        
        In original version we computed Jacobian first and then (maybe not soon)
        composed it with passed gradient to get the influence of each entry
        in this particular tensor on the final value, that is a scalar.
        
        Here we pass the first step (i.e. Jacobian computation) and perform 
        composition in place (i.e. compute needed entries of Jacobian in place).
        
        Odd indexes you see in the following loop are motivated by flexibility
        of numpy's ndarray binary operations.
        """
        composition = np.zeros_like(child.value, dtype=float)
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                composition[i % composition.shape[0]][j % composition.shape[1]] += 1 * grad[i][j]
        child.propagate_(child, composition)


def sub_propagate(tensor, grad):
    """sub_propagate"""
    if not tensor.c_g_:
        return

    # first child
    if tensor.children_[0] is not None:
        composition = np.zeros_like(tensor.children_[0].value, dtype=float)
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                composition[i % composition.shape[0]][j % composition.shape[1]] += 1 * grad[i][j]
        tensor.children_[0].propagate_(tensor.children_[0], composition)

    # second child
    if tensor.children_[1] is not None:
        composition = np.zeros_like(tensor.children_[1].value, dtype=float)
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                composition[i % composition.shape[0]][j % composition.shape[1]] += -1 * grad[i][j]
        tensor.children_[1].propagate_(tensor.children_[1], composition)


def sum_propagate(tensor, grad):
    """sum_propagate"""
    if not tensor.c_g_:
        return

    tensor.children_[0].propagate_(tensor.children_[0], np.ones_like(tensor.children_[0].value) * grad)


def neg_propagate(tensor, grad):
    """neg_propagate"""
    if not tensor.c_g_:
        return

    tensor.children_[0].propagate_(tensor.children_[0], -1 * grad)


def mul_propagate(tensor, grad):
    """mul_propagate"""
    if not tensor.c_g_:
        return

    c1 = tensor.children_[0]
    c2 = tensor.children_[1]
    if c1.c_g_:
        com_1 = np.zeros_like(c1.value, dtype=float)
    if c2.c_g_:
        com_2 = np.zeros_like(c2.value, dtype=float)

    for i in range(grad.shape[0]):
        for j in range(grad.shape[1]):
            if c1.c_g_:
                com_1[i % com_1.shape[0]][j % com_1.shape[1]] += \
                    c2.value[i % c2.value.shape[0]][j % c2.value.shape[1]] * grad[i][j]
            if c2.c_g_:
                com_2[i % com_2.shape[0]][j % com_2.shape[1]] += \
                    c1.value[i % c1.value.shape[0]][j % c1.value.shape[1]] * grad[i][j]

    if c1.c_g_:
        c1.propagate_(c1, com_1)
    if c2.c_g_:
        c2.propagate_(c2, com_2)


def truediv_propagate(tensor, grad):
    """truediv_propagate"""
    if not tensor.c_g_:
        return

    c1 = tensor.children_[0]
    c2 = tensor.children_[1]
    if c1.c_g_:
        com_1 = np.zeros_like(c1.value, dtype=float)
    if c2.c_g_:
        com_2 = np.zeros_like(c2.value, dtype=float)

    for i in range(grad.shape[0]):
        for j in range(grad.shape[1]):
            a = c1.value[i % c1.value.shape[0]][j % c1.value.shape[1]]
            b = c2.value[i % c2.value.shape[0]][j % c2.value.shape[1]]
            if c1.c_g_:
                com_1[i % com_1.shape[0]][j % com_1.shape[1]] += grad[i][j] * 1 / b
            if c2.c_g_:
                com_2[i % com_2.shape[0]][j % com_2.shape[1]] += grad[i][j] * -a / b**2

    if c1.c_g_:
        c1.propagate_(c1, com_1)
    if c2.c_g_:
        c2.propagate_(c2, com_2)


def matmul_propagate(tensor, grad):
    """matmul_propagate"""
    if not tensor.c_g_:
        return

    c1 = tensor.children_[0]
    c2 = tensor.children_[1]
    if c1.c_g_:
        com_1 = np.zeros_like(c1.value, dtype=float)
    if c2.c_g_:
        com_2 = np.zeros_like(c2.value, dtype=float)

    for i in range(grad.shape[0]):
        for j in range(grad.shape[1]):
            for k in range(c1.value.shape[1]):
                if c1.c_g_:
                    com_1[i][k] += grad[i][j] * c2.value[k][j]
                if c2.c_g_:
                    com_2[k][j] += grad[i][j] * c1.value[i][k]

    if c1.c_g_:
        c1.propagate_(c1, com_1)
    if c2.c_g_:
        c2.propagate_(c2, com_2)


def pow_propagate(tensor, grad):
    """pow_propagate"""
    if not tensor.c_g_:
        return

    c1 = tensor.children_[0]
    c2 = tensor.children_[1]
    if c1.c_g_:
        com_1 = np.zeros_like(c1.value, dtype=float)
    if c2.c_g_:
        com_2 = np.zeros_like(c2.value, dtype=float)

    for i in range(tensor.value.shape[0]):
        for j in range(tensor.value.shape[1]):
            a = c1.value[i % c1.value.shape[0]][j % c1.value.shape[1]]
            b = c2.value[i % c2.value.shape[0]][j % c2.value.shape[1]]
            if c1.c_g_:
                com_1[i % com_1.shape[0]][j % com_1.shape[1]] += b * a ** (b - 1)
            if c2.c_g_:
                com_2[i % com_2.shape[0]][j % com_2.shape[1]] += a ** b * math.log(a)

    if c1.c_g_:
        c1.propagate_(c1, com_1)
    if c2.c_g_:
        c2.propagate_(c2, com_2)


# Core type

class Tensor:
    def __init__(self, value: np.ndarray, compute_g=False):
        if value.size == 0:
            raise ValueError('Cannot initialize the tensor with empty value.')
        elif value.size == 1:
            self.value = value.reshape((1, 1))
        elif len(value.shape) == 1:
            self.value = value.reshape((value.shape[0], 1))
        elif len(value.shape) == 2:
            self.value = value
        else:
            raise ValueError('Initialization with array of > 2 dimensions if forbidden.')

        self.c_g_ = compute_g
        self.is_leaf = True
        self.grad = None
        self.propagate_ = leaf_accumulator # gradient back-propagator, gradient accumulator by default
        self.children_ = ()

    def __str__(self):
        return str(self.value)

    def back_prop(self):
        if not self.c_g_:
            return

        if self.value.size != 1:
            raise ValueError('Cannot back-propagate gradient from tensor that is not a scalar.')

        # cleaning previous gradients
        def clean_grads(t: Tensor):
            if t.c_g_:
                t.grad = None
                for child in t.children_:
                    if child is not None:
                        clean_grads(child)

        clean_grads(self)
        self.propagate_(self, np.array([[1]]))

    def __add__(self, other):
        n_t = Tensor(self.value + other.value, self.c_g_ | other.c_g_)
        n_t.is_leaf = False
        if n_t.c_g_:
            n_t.propagate_ = add_propagate
            n_t.children_ = (self if self.c_g_ else None, other if other.c_g_ else None)
        return n_t

    def __sub__(self, other):
        n_t = Tensor(self.value - other.value, self.c_g_ | other.c_g_)
        n_t.is_leaf = False
        if n_t.c_g_:
            n_t.propagate_ = sub_propagate
            n_t.children_ = (self if self.c_g_ else None, other if other.c_g_ else None)
        return n_t

    def __neg__(self):
        n_t = Tensor(-self.value, self.c_g_)
        n_t.is_leaf = False
        if n_t.c_g_:
            n_t.propagate_ = neg_propagate
            n_t.children_ = (self, )
        return n_t

    def __mul__(self, other):
        n_t = Tensor(self.value * other.value, self.c_g_ | other.c_g_)
        n_t.is_leaf = False
        if n_t.c_g_:
            n_t.propagate_ = mul_propagate
            # to back-propagate through this operation we must know both children
            n_t.children_ = (self, other)
        return n_t

    def __truediv__(self, other):
        n_t = Tensor(self.value / other.value, self.c_g_ | other.c_g_)
        n_t.is_leaf = False
        if n_t.c_g_:
            n_t.propagate_ = truediv_propagate
            n_t.children_ = (self if self.c_g_ else None, other if other.c_g_ else None)
        return n_t

    def __matmul__(self, other):
        n_t = Tensor(self.value @ other.value, self.c_g_ | other.c_g_)
        n_t.is_leaf = False
        if n_t.c_g_:
            n_t.propagate_ = matmul_propagate
            n_t.children_ = (self if self.c_g_ else None, other if other.c_g_ else None)
        return n_t

    def __pow__(self, other):
        n_t = Tensor(self.value ** other.value, self.c_g_ | other.c_g_)
        n_t.is_leaf = False
        if n_t.c_g_:
            n_t.propagate_ = pow_propagate
            n_t.children_ = (self if self.c_g_ else None, other if other.c_g_ else None)
        return n_t

    def sum(self):
        n_t = Tensor(self.value.sum(), self.c_g_)
        n_t.is_leaf = False
        if n_t.c_g_:
            n_t.propagate_ = sum_propagate
            n_t.children_ = (self,)
        return n_t





