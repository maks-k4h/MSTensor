import numpy as np


class Tensor:
    """
    Tensor

    Class Tensor represents multidimensional arrays and
    supports auto-differentiation harnessing back-propagation
    algorithm."""

    # Due to (my) simple convention, all tensors are matrices (2d arrays):
    # scalars are 1 by 1 matrices, vectors are n by 1 matrices, matrices
    # are represented respectively and higher dimensional arrays are not
    # supported.
    # All partial derivatives are 4-dimensional tensors built on principle:
    # for every output every input.

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
        self.grad = None
        self.parents_ = ()  # stores pairs: (parent, partial_derivative_with_respect_to_parent)

    def __str__(self):
        return str(self.value)

    def back_prop(self):
        if not self.c_g_:
            return
        if self.grad is None:  # check if it's a leaf (back_prop was initially called on it)
            assert self.value.size == 1  # leaf must be a scalar
            self.grad = np.ones_like(self.value)

        # composes Jacobian and gradient
        def compose(jac: np.ndarray, grad: np.ndarray):
            for i in range(jac.shape[0]):
                for j in range(jac.shape[1]):
                    jac[i][j] *= grad[i][j]
            return jac.sum(axis=0).sum(axis=0)

        for parent in self.parents_:
            if parent[0] and parent[0].c_g_:
                if not parent[0].grad:
                    parent[0].grad = np.zeros_like(parent[0].value, dtype=float)

                # debugging purposes
                assert len(parent[1].shape) == 4

                parent[0].grad += compose(parent[1], self.grad)
                parent[0].back_prop()

    def __add__(self, other):
        n_t = Tensor(self.value + other.value, compute_g=self.c_g_ or other.c_g_)
        if n_t.c_g_:
            g1 = np.zeros(shape=n_t.value.shape + self.value.shape)
            g2 = np.zeros(shape=n_t.value.shape + other.value.shape)
            for i in range(n_t.value.shape[0]):
                for j in range(n_t.value.shape[1]):
                    g1[i][j][i % g1.shape[2]][j % g1.shape[3]] = 1
                    g2[i][j][i % g2.shape[2]][j % g2.shape[3]] = 1
            n_t.parents_ = ((self, g1), (other, g2))
        return n_t

    def __sub__(self, other):
        n_t = Tensor(self.value - other.value, compute_g=self.c_g_ or other.c_g_)
        if n_t.c_g_:
            g1 = np.zeros(shape=n_t.value.shape + self.value.shape)
            g2 = np.zeros(shape=n_t.value.shape + other.value.shape)
            for i in range(n_t.value.shape[0]):
                for j in range(n_t.value.shape[1]):
                    g1[i][j][i % g1.shape[2]][j % g1.shape[3]] = 1
                    g2[i][j][i % g2.shape[2]][j % g2.shape[3]] = -1
            n_t.parents_ = ((self, g1), (other, g2))
        return n_t

    def __pow__(self, power):
        n_t = Tensor(self.value ** power, compute_g=self.c_g_)
        if n_t.c_g_:
            g = np.zeros(shape=n_t.value.shape + self.value.shape)
            for i in range(n_t.value.shape[0]):
                for j in range(n_t.value.shape[1]):
                    g[i][j][i][j] += power * self.value[i][j] ** (power - 1)
            n_t.parents_ = ((self, g),)
        return n_t

    def __matmul__(self, other):
        n_t = Tensor(self.value @ other.value, compute_g=(self.c_g_ or other.c_g_))
        if n_t.c_g_:
            g1 = np.zeros(shape=n_t.value.shape + self.value.shape)
            g2 = np.zeros(shape=n_t.value.shape + other.value.shape)
            for i in range(n_t.value.shape[0]):
                for j in range(n_t.value.shape[1]):
                    for k in range(self.value.shape[1]):
                        g1[i][j][i][k] += other.value[k][j]
                        g2[i][j][k][j] += self.value[i][k]
            n_t.parents_ = ((self, g1), (other, g2))
        return n_t

    def __mul__(self, other):
        n_t = Tensor(self.value * other.value, compute_g=self.c_g_ or other.c_g_)
        if n_t.c_g_:
            g1 = np.zeros(shape=n_t.value.shape + self.value.shape)
            g2 = np.zeros(shape=n_t.value.shape + other.value.shape)
            for i in range(n_t.value.shape[0]):
                for j in range(n_t.value.shape[1]):
                    g1[i][j][i % g1.shape[2]][j % g1.shape[3]] = other.value[i % g2.shape[2]][j % g2.shape[3]]
                    g2[i][j][i % g2.shape[2]][j % g2.shape[3]] = self.value[i % g1.shape[2]][j % g1.shape[3]]
            n_t.parents_ = ((self, g1), (other, g2))
        return n_t

    def __truediv__(self, other):
        n_t = Tensor(self.value / other.value, compute_g=self.c_g_ or other.c_g_)
        if n_t.c_g_:
            g1 = np.zeros(shape=n_t.value.shape + self.value.shape)
            g2 = np.zeros(shape=n_t.value.shape + other.value.shape)
            for i in range(n_t.value.shape[0]):
                for j in range(n_t.value.shape[1]):
                    g1[i][j][i % g1.shape[2]][j % g1.shape[3]] = 1 / other.value[i % g2.shape[2]][j % g2.shape[3]]
                    g2[i][j][i % g2.shape[2]][j % g2.shape[3]] = -self.value[i % g1.shape[2]][j % g1.shape[3]] / \
                                                                 other.value[i % g2.shape[2]][j % g2.shape[3]] ** 2
            n_t.parents_ = ((self, g1), (other, g2))
        return n_t

    def sum(self):
        n_t = Tensor(np.array([self.value.sum()]), self.c_g_)
        if n_t.c_g_:
            n_t.parents_ = ((self, np.ones((1, 1) + self.value.shape)),)
        return n_t
