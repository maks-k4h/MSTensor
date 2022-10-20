from MSTensor import *
import math


def initialization_test():
    """
    Tensor Initialization Test

    Conducting simple initializations and checking their
    correctness. Initialization of tensors with arrays
    of forbidden dimensionality is also hold, exception
    raising checked.

    """

    # initializing with empty value — exception expected
    try:
        t = Tensor(np.array([]))
    except:
        ...
    else:
        raise AssertionError('Tensor initialization with empty value succeeded.')

    t = Tensor(np.array([1]))
    assert t.value.size == 1
    assert len(t.value.shape) == 2
    assert not t.grad
    assert not t.c_g_
    assert len(t.parents_) == 0

    t = Tensor(np.array([1, 2]))
    assert t.value.size == 2
    assert t.value.shape == (2, 1)
    assert not t.grad
    assert not t.c_g_
    assert len(t.parents_) == 0

    t = Tensor(np.array([[1, 2], [2, 1]]))
    assert t.value.size == 4
    assert len(t.value.shape) == 2
    assert not t.grad
    assert not t.c_g_
    assert len(t.parents_) == 0

    # initializing with high-dimensional array (d > 2) — exception expected
    try:
        t = Tensor(np.ones(shape=(2, 2, 2)))
    except:
        ...
    else:
        raise AssertionError('Initialization of the tensor with'
                             'high-dimensional array succeeded.')

    t = Tensor(np.array([1]), compute_g=True)
    assert t.value.size == 1
    assert len(t.value.shape) == 2
    assert not t.grad
    assert t.c_g_
    assert len(t.parents_) == 0

    print("Tensor Initialization tests succeeded.")


def addition_test():
    """
    Tensor Addition Test

    Performing addition of different forms, computing the gradient
    of the operation with respect to terms, checking attributes,
    conducting forbidden additions and checking if the proper error
    was raised.
    """

    t1 = Tensor(np.array([2]))
    t2 = Tensor(np.array([13]))
    res = t1 + t2
    assert res.value.size == 1
    assert res.value == np.array([[15]])
    assert not res.c_g_
    assert not res.grad

    a = Tensor(np.array([1]), True)
    b = Tensor(np.array([1]), False)
    assert (a + b).c_g_ and (b + a).c_g_
    # we don't check for parents here as it depends on realization

    t1 = Tensor(np.array([2]), True)
    t2 = Tensor(np.array([13]), True)
    res = t1 + t2
    assert res.value.size == 1
    assert res.value == np.array([[15]])
    assert res.c_g_
    assert not res.grad
    assert len(res.parents_) == 2
    res.back_prop()
    assert t1.grad == np.array([[1]])
    assert t2.grad == np.array([[1]])

    t1 = Tensor(np.array([1]), True)
    t2 = Tensor(np.array([2]), True)
    t3 = Tensor(np.array([3]), True)
    res = t1 + t2 + t3
    assert res.value[0][0] == 6
    res.back_prop()
    assert t1.grad[0][0] == 1
    assert t2.grad[0][0] == 1
    assert t3.grad[0][0] == 1

    t1 = Tensor(np.array([[1, 2], [1, 3]]))
    t2 = Tensor(np.array([[1, -1], [1, -1]]))
    t3 = Tensor(np.array([1, -1]))  # must be interpreted as vector-column
    t4 = Tensor(np.array([10]))
    a1 = t1 + t2
    assert a1.value[0][0] == 2 and a1.value[0][1] == 1 and \
           a1.value[1][0] == 2 and a1.value[1][1] == 2
    a2 = a1 + t3
    assert a2.value[0][0] == 3 and a2.value[0][1] == 2 and \
           a2.value[1][0] == 1 and a2.value[1][1] == 1
    a3 = a2 + t4
    assert a3.value[0][0] == 13 and a3.value[0][1] == 12 and \
           a3.value[1][0] == 11 and a3.value[1][1] == 11

    # forbidden additions
    t1 = Tensor(np.array([[1, 2, 3], [1, 3, 3]]))
    t2 = Tensor(np.array([[1, -1], [1, -1]]))
    t3 = Tensor(np.array([1, 2, 3]))
    t4 = Tensor(np.array([[1, 2, 3]]))
    try:
        t1 + t2
    except:
        ...
    else:
        raise ArithmeticError('Tensors of inappropriate dimensions were added!')
    try:
        t1 + t3  # must fail since we cannot add tensors with such dims: (3,1) + (2,3)
    except:
        ...
    else:
        raise ArithmeticError('Tensors of inappropriate dimensions were added!')
    try:
        t1 + t4  # must success since we  add tensors with such dims: (1,3) + (2,3)
    except:
        raise ArithmeticError('Tensors of right dimensions were not added!')

    # Since the gradient of an operation with result with more than one
    # entry cannot be auto-computed directly, so we only tests if assertion
    # exception raised on such a try.
    t1 = Tensor(np.array([1, 2]), True)
    t2 = Tensor(np.array([1, 0]), True)
    a = t1 + t2
    assert a.c_g_
    try:
        a.back_prop()
    except:
        ...
    else:
        raise NotImplementedError('Back Propagation from a tensor that is not a '
                                  'scalar succeeded.')

    print('Tensor Addition Tests Succeeded.')


def subtraction_test():
    """
    Tensor Addition Test

    Performing subtractions on tensors different forms, computing the
    gradient of the operation with respect to terms, checking derived
    attributes, conducting forbidden subtractions and checking if the
    proper error is raised.
    """

    t1 = Tensor(np.array([2]))
    t2 = Tensor(np.array([13]))
    res = t1 + t2
    assert res.value.size == 1
    assert res.value == np.array([[15]])
    assert not res.c_g_
    assert not res.grad

    a = Tensor(np.array([1]), True)
    b = Tensor(np.array([1]), False)
    assert (a + b).c_g_ and (b + a).c_g_
    # we don't check for parents here as it depends on realization

    t1 = Tensor(np.array([20]), True)
    t2 = Tensor(np.array([13]), True)
    res = t1 - t2
    assert res.value.size == 1
    assert res.value == np.array([[7]])
    assert res.c_g_
    assert not res.grad
    assert len(res.parents_) == 2
    res.back_prop()
    assert t1.grad == np.array([[1]])
    assert t2.grad == np.array([[-1]])

    t1 = Tensor(np.array([3]), True)
    t2 = Tensor(np.array([2]), True)
    t3 = Tensor(np.array([1]), True)
    res = t1 - t2 - t3
    assert res.value[0][0] == 0
    res.back_prop()
    assert t1.grad[0][0] == 1
    assert t2.grad[0][0] == -1
    assert t3.grad[0][0] == -1

    t1 = Tensor(np.array([[1, 2], [1, 3]]))
    t2 = Tensor(np.array([[1, -1], [1, -1]]))
    t3 = Tensor(np.array([1, -1]))  # must be interpreted as vector-column
    t4 = Tensor(np.array([-1]))
    a1 = t1 - t2
    assert a1.value[0][0] == 0 and a1.value[0][1] == 3 and \
           a1.value[1][0] == 0 and a1.value[1][1] == 4
    a2 = a1 - t3
    assert a2.value[0][0] == -1 and a2.value[0][1] == 2 and \
           a2.value[1][0] == 1 and a2.value[1][1] == 5
    a3 = a2 - t4
    assert a3.value[0][0] == 0 and a3.value[0][1] == 3 and \
           a3.value[1][0] == 2 and a3.value[1][1] == 6

    # forbidden additions
    t1 = Tensor(np.array([[1, 2, 3], [1, 3, 3]]))
    t2 = Tensor(np.array([[1, -1], [1, -1]]))
    t3 = Tensor(np.array([1, 2, 3]))
    t4 = Tensor(np.array([[1, 2, 3]]))
    try:
        t1 - t2
    except:
        ...
    else:
        raise ArithmeticError('Subtraction on tensors of wrong dimensions was performed!')
    try:
        t1 - t3  # must fail since we cannot subtract tensors with dims: (3,1) and (2,3)
    except:
        ...
    else:
        raise ArithmeticError('Subtraction on tensors of wrong dimensions was performed!')
    try:
        t1 - t4  # must success since we subtract tensors with dims: (1,3) and (2,3)
    except:
        raise ArithmeticError('Tensors of right dimensions were not subtracted!')

    # Since the gradient of an operation with result with more than one
    # entry cannot be auto-computed directly, so we only tests if assertion
    # exception raised on such a try.
    t1 = Tensor(np.array([1, 2]), True)
    t2 = Tensor(np.array([1, 0]), True)
    a = t1 - t2
    assert a.c_g_
    try:
        a.back_prop()
    except:
        ...
    else:
        raise NotImplementedError('Back Propagation from a tensor that is not a '
                                  'scalar succeeded.')

    print('Tensor Addition Tests Succeeded.')


def multiplication_test():
    """
    Tensor Multiplication Tests

    Performing multiplication of different forms, computing the gradient
    of the operation with respect to terms (when possible), checking
    attributes, conducting forbidden multiplication and checking if the
    proper error was raised.
    """

    t1 = Tensor(np.array([2]))
    t2 = Tensor(np.array([11]))
    res = t1 * t2
    assert res.value.size == 1
    assert res.value == np.array([[22]])
    assert not res.c_g_
    assert not res.grad

    a = Tensor(np.array([1]), True)
    b = Tensor(np.array([1]), False)
    assert (a * b).c_g_ and (b * a).c_g_
    # we don't check for parents here as it depends on realization

    t1 = Tensor(np.array([2]), True)
    t2 = Tensor(np.array([11]), True)
    res = t1 * t2
    assert res.value.size == 1
    assert res.value == np.array([[22]])
    assert res.c_g_
    assert not res.grad
    assert len(res.parents_) == 2
    res.back_prop()
    assert t1.grad == np.array([[11]])
    assert t2.grad == np.array([[2]])

    t1 = Tensor(np.array([1]), True)
    t2 = Tensor(np.array([2]), True)
    t3 = Tensor(np.array([3]), True)
    res = t1 * t2 * t3
    assert res.value[0][0] == 6
    res.back_prop()
    assert t1.grad[0][0] == 6
    assert t2.grad[0][0] == 3
    assert t3.grad[0][0] == 2

    t1 = Tensor(np.array([[1, 2], [1, 3]]))
    t2 = Tensor(np.array([[1, 2], [1, 0]]))
    t3 = Tensor(np.array([3, 2]))  # must be interpreted as vector-column
    t4 = Tensor(np.array([11]))
    a1 = t1 * t2
    assert a1.value[0][0] == 1 and a1.value[0][1] == 4 and \
           a1.value[1][0] == 1 and a1.value[1][1] == 0
    a2 = a1 * t3
    assert a2.value[0][0] == 3 and a2.value[0][1] == 12 and \
           a2.value[1][0] == 2 and a2.value[1][1] == 0
    a3 = a2 * t4
    assert a3.value[0][0] == 33 and a3.value[0][1] == 132 and \
           a3.value[1][0] == 22 and a3.value[1][1] == 0

    # forbidden additions
    t1 = Tensor(np.array([[1, 2, 3], [1, 3, 3]]))
    t2 = Tensor(np.array([[1, -1], [1, -1]]))
    t3 = Tensor(np.array([1, 2, 3]))
    t4 = Tensor(np.array([[1, 2, 3]]))
    try:
        t1 * t2
    except:
        ...
    else:
        raise ArithmeticError('Tensors of inappropriate dimensions were multiplied!')
    try:
        t1 + t3  # must fail since we cannot multiply tensors with dims (3,1) and (2,3)
    except:
        ...
    else:
        raise ArithmeticError('Tensors of inappropriate dimensions were multiplied!')
    try:
        t1 + t4  # must success since we multiply tensors with dims (1,3) and (2,3)
    except:
        raise ArithmeticError('Tensors of right dimensions were not multiplied!')

    # Since the gradient of an operation with result with more than one
    # entry cannot be auto-computed directly, so we only tests if assertion
    # exception raised on such a try.
    t1 = Tensor(np.array([1, 2]), True)
    t2 = Tensor(np.array([1, 0]), True)
    a = t1 * t2
    assert a.c_g_
    try:
        a.back_prop()
    except:
        ...
    else:
        raise NotImplementedError('Back Propagation from a tensor that is not a '
                                  'scalar succeeded.')

    print('Tensor Multiplication Tests Succeeded.')


def truediv_test():
    """
    Tensor Division (truediv) Test

    The test includes dividing tensors by tensors representing scalars,
    vectors and matrices, division of matrices over vectors and scalars,
    vector over scalars, etc. Handling of tries of division of tensors with
    inappropriate dimensions is also checked.
    """

    t1 = Tensor(np.array([2]), True)
    t2 = Tensor(np.array([5]), False)
    assert (t1 / t2).c_g_ and (t2 / t1).c_g_
    t2.c_g_ = True
    a = t1 / t2
    a.back_prop()
    assert a.value.shape == (1, 1) and a.value[0][0] == 2 / 5
    assert t1.grad.shape == (1, 1) and t1.grad[0][0] == 1 / 5
    assert t2.grad.shape == (1, 1) and t2.grad[0][0] == - 2 / 25

    # vector over scalar
    t1 = Tensor(np.array([3, 7, 2]), True)
    t2 = Tensor(np.array([2]), True)
    a = (t1 / t2).sum()
    a.back_prop()
    assert t1.grad.shape == (3, 1)
    assert t1.grad[0][0] == t1.grad[1][0] == t1.grad[2][0] == 1/2
    assert t2.grad.shape == (1, 1)
    assert t2.grad[0][0] == -3/4 -7/4 -2/4

    # scalar over vector (numpy allows, we too)
    t1 = Tensor(np.array([3, 7, 2]), True)
    t2 = Tensor(np.array([2]), True)
    a = (t2 / t1).sum()
    a.back_prop()
    assert t1.grad.shape == (3, 1)
    assert t1.grad[0][0] == -2 / 3**2
    assert t1.grad[1][0] == -2 / 7**2
    assert t1.grad[2][0] == -2 / 2**2
    assert t2.grad.shape == (1, 1)
    assert t2.grad[0][0] == 1/3 + 1/7 + 1/2

    # matrix over vector
    t1 = Tensor(np.array([[1, 7], [3, 5]]), True)
    t2 = Tensor(np.array([9, 2]), True)
    a = (t1 / t2).sum()
    a.back_prop()
    assert a.value[0][0] == 1/9 + 7/9 + 3/2 + 5/2
    assert t1.grad.shape == (2, 2)
    assert t1.grad[0][0] == t1.grad[0][1] == 1/9
    assert t1.grad[1][0] == t1.grad[1][1] == 1/2
    assert t2.grad.shape == (2, 1)
    assert t2.grad[0][0] == -(1 + 7)/9**2
    assert t2.grad[1][0] == -(3 + 5)/2**2

    # matrix over matrix
    t1 = Tensor(np.array([[2, 3], [4, 5]]), True)
    t2 = Tensor(np.array([[9, 2], [8, 7]]), True)
    a = (t1 / t2).sum()
    a.back_prop()
    assert t1.grad.shape == t2.grad.shape == (2, 2)
    assert t1.grad[0][0] == 1/9
    assert t1.grad[0][1] == 1/2
    assert t1.grad[1][0] == 1/8
    assert t1.grad[1][1] == 1/7
    assert t2.grad[0][0] == -2/9**2
    assert t2.grad[0][1] == -3/2**2
    assert t2.grad[1][0] == -4/8**2
    assert t2.grad[1][1] == -5/7**2

    # inappropriate dimensions
    t1 = Tensor(np.array([[2, 3, 5], [4, 5, 9]]), True)
    t2 = Tensor(np.array([[9, 2], [8, 7]]), True)
    try:
        t1 / t2
    except:
        ...
    else:
        raise ArithmeticError('Division (truediv) of tensors with inappropriate dimensions'
                              ' succeeded.')

    print('Tensor Division (truediv) Tests Succeeded.')


def matmul_test():
    """
    Tensor Matrix Multiplication Test

    The test contains usage of matmul to compute simple scalar product and its gradient,
    try to perform multiplication of vector by scalar and other version of matmul
    where wrong dimensions are involved. We also compute the dot product between vectors
    and its gradient. Matrices' multiplication included, without gradients though
    (ones more, we cannot start to back-propagate gradient from a tensor that is not
    a scalar, e.g. matrix).
    """

    # matmul as mul for scalars and the gradient
    t1 = Tensor(np.array([5]))
    t2 = Tensor(np.array([3]))
    a = t1 @ t2
    assert not a.c_g_ and not a.grad
    assert a.value.size == 1
    assert a.value.shape == (1, 1)
    assert a.value == np.array([[15]])

    a = Tensor(np.array([1]), True)
    b = Tensor(np.array([1]), False)
    assert (a @ b).c_g_ and (b @ a).c_g_

    t1 = Tensor(np.array([5]), True)
    t2 = Tensor(np.array([3]), True)
    a = t1 @ t2
    assert a.c_g_ and not a.grad
    assert a.value.size == 1
    assert a.value.shape == (1, 1)
    assert a.value == np.array([[15]])
    a.back_prop()
    assert t1.grad.size == 1 and t1.grad.shape == (1, 1)
    assert t1.grad == np.array([[3]])
    assert t2.grad.size == 1 and t2.grad.shape == (1, 1)
    assert t2.grad == np.array([[5]])

    # matmul does not work for scalar times vector (and others wrongs)
    t1 = Tensor(np.array([5]))
    t2 = Tensor(np.array([3, 4, 5]))
    try:
        a = t1 @ t2
    except:
        ...
    else:
        raise ArithmeticError('The matmul of a vector and a scalar was computed.')

    t1 = Tensor(np.array([[5, 4]]))
    t2 = Tensor(np.array([3, 4, 5]))
    try:
        a = t1 @ t2
    except:
        ...
    else:
        raise ArithmeticError('The dot product of vectors of different dimensions'
                              ' was computed.')

    t1 = Tensor(np.array([[5, 4],
                          [2, 3]]))
    t2 = Tensor(np.array([[3, 4],
                          [1, 1],
                          [2, 3]]))
    try:
        a = t1 @ t2
    except:
        ...
    else:
        raise ArithmeticError('The dot product of matrices of inappropriate dimensions'
                              ' was computed.')

    # matmul to compute dot product between vectors and the gradient
    t1 = Tensor(np.array([[1, 2, 3]]), True)  # row
    t2 = Tensor(np.array([2, 2, 2]), True)    # column
    a = t1 @ t2
    assert a.value.size == 1 and a.value.shape == (1, 1)
    assert a.value == np.array([[12]])
    a.back_prop()
    assert t1.grad.shape == (1, 3)
    assert t2.grad.shape == (3, 1)
    assert t1.grad[0][0] == 2 and t1.grad[0][1] == 2 and t1.grad[0][2] == 2
    assert t2.grad[0][0] == 1 and t2.grad[1][0] == 2 and t2.grad[2][0] == 3

    # matmul to compute matrix multiplication between matrices (simple test)
    m1 = np.array([[1, 2, 3],
                   [6, 5, 4]])
    m2 = np.array([[9, 5, 5],
                   [5, 6, 2],
                   [2, 9, 1]])
    t1 = Tensor(m1)
    t2 = Tensor(m2)
    a = t1 @ t2
    t = m1 @ m2
    assert not a.c_g_ and not a.grad
    for i in range(2):
        for j in range(3):
            assert a.value[i][j] == t[i][j]

    print('Tensor Matrix Multiplication Tests Succeeded.')


def pow_test():
    """
    Tensor Power Tests

    Test raising tensors of different forms to different powers, including
    special cases of power 0 and negative power of tensors with 0 entries.
    We also check if attributes are passed to children and take the
    derivative of the operation.
    """

    # one-dimensional tensors
    a = Tensor(np.array([5]))
    b = Tensor(np.array([2]))
    r = a ** b
    assert not r.c_g_
    assert r.value[0][0] == 25

    a = Tensor(np.array([5]), True)
    b = Tensor(np.array([2]), True)
    r = a ** b
    r.back_prop()
    assert a.c_g_
    assert r.value[0][0] == 25
    assert a.grad == 10
    assert b.grad == 5**2 * math.log(5)

    a = Tensor(np.array([5]), True)
    b = Tensor(np.ones(1), True)
    r = a ** b
    r.back_prop()
    assert r.c_g_
    assert r.value[0][0] == 5
    assert a.grad == 1
    assert b.grad == 5 * math.log(5)

    a = Tensor(np.array([5.]), True)
    b = Tensor(np.array([0.]), True)
    r = a ** b
    r.back_prop()
    assert r.c_g_
    assert r.value[0][0] == 1
    assert a.grad == 0
    assert b.grad == math.log(5)

    a = Tensor(np.array([5.]), True)
    b = Tensor(np.array([-1.]), True)
    r = a ** b
    r.back_prop()
    assert r.c_g_
    assert r.value[0][0] == 1/5
    assert a.grad == -1/25
    assert b.grad - math.log(5) / 5 < 1e-6

    # higher-dimensional tensors

    a = Tensor(np.array([5., 5.]), True)
    b = Tensor(np.array([-1.]), True)
    r = (a ** b).sum()
    r.back_prop()
    assert r.c_g_
    assert r.value[0][0] == 2/5
    assert a.grad.shape == (2, 1)
    assert a.grad[0][0] == a.grad[1][0] == -1/25
    assert b.grad.shape == (1, 1)
    assert b.grad[0][0] - 2 * math.log(5) / 5 < 1e-6

    a = Tensor(np.array([5.]), True)
    b = Tensor(np.array([-1., -1.]), True)
    r = (a ** b).sum()
    r.back_prop()
    assert r.value[0][0] == 2 / 5
    assert a.grad.shape == (1, 1)
    assert a.grad[0][0] == - 2 / 5 ** 2
    assert b.grad.shape == (2, 1)
    assert b.grad[0][0] - 2 * math.log(5) / 5 < 1e-6
    assert b.grad[1][0] - 2 * math.log(5) / 5 < 1e-6

    a = Tensor(np.array([[1., 2.], [3., 4.]]), True)
    b = Tensor(np.array([2, 4]), True)
    r = (a ** b).sum()
    r.back_prop()
    assert r.value == 1**2 + 2**2 + 3**4 + 4**4
    assert a.grad.shape == (2, 2)
    assert b.grad.shape == (2, 1)
    assert a.grad[0][0] == 2 * 1**1
    assert a.grad[0][1] == 2 * 2**1
    assert a.grad[1][0] == 4 * 3**3
    assert a.grad[1][1] == 4 * 4**3
    assert b.grad[0][0] - (1**2 * math.log(1) + 2**2 * math.log(2)) < 1e-6
    assert b.grad[1][0] - (3**4 * math.log(3) + 4**4 * math.log(4)) < 1e-6


    print('Tensor Power Tests Succeeded.')



def sum_test():
    """
    Tensor Sum Method Test

    Calls sum on several tensors and computes their gradients (that are just ones).
    """

    t = Tensor(np.array([121]))
    a = t.sum()
    assert not a.grad and not a.c_g_
    assert a.value.size == 1 and a.value[0][0] == 121

    t = Tensor(np.array([121]), True)
    a = t.sum()
    a.back_prop()
    assert a.grad and a.c_g_
    assert t.grad == np.array([[1]])

    t = Tensor(np.array([1, 2, 3, 4]), True)
    a = t.sum()
    a.back_prop()
    assert a.value == 10
    assert t.grad.shape == (4, 1)
    assert t.grad[0][0] == t.grad[1][0] == t.grad[2][0] == t.grad[3][0] == 1

    t = Tensor(np.random.rand(5, 10), True)
    a = t.sum()
    a.back_prop()
    assert t.grad.shape == (5, 10)
    for row in t.grad:
        for elem in row:
            assert elem == 1

    print('Tensor Sum Method Test Succeeded.')


def chain_rule_test():
    """
    Tensor Chain Rule (Back Propagation) Test

    Conduct back propagation first on simple chain with scalars, then on chain with
    scalar branches (e.g. a, b depend on k and then c depends on both a, b). Afterwards,
    chains involving both variables with gradients that must be computed and not are
    build and tensors that were not expected to get gradients are checked if they did.
    Then we also used autodiff on several simple examples with matrices.
    """

    # Simple chain rule with scalars
    a = Tensor(np.array([5]), True)
    b = Tensor(np.array([3]), True)
    c = Tensor(np.array([10]), True)
    z = a * b + c
    assert z.c_g_
    z.back_prop()
    assert z.value[0][0] == 25
    assert a.grad == np.array([[3]])
    assert b.grad == np.array([[5]])
    assert c.grad == np.array([[1]])

    # Chain rule with scalars and branches (e.g. a, b depend on k
    # and then c depends on both a, b)
    a = Tensor(np.array([5]), True)
    b = Tensor(np.array([3]), True)
    c = a * b
    r = a + c
    r.back_prop()
    assert r.value[0][0] == 20
    assert c.grad.shape == (1, 1) and c.grad[0][0] == 1
    assert b.grad.shape == (1, 1) and b.grad[0][0] == 5
    assert a.grad.shape == (1, 1) and a.grad[0][0] == 3 + 1

    a = Tensor(np.array([3]), True)
    r = a * Tensor(np.array(4)) + a
    r.back_prop()
    assert a.grad[0][0] == 5

    # Chains involving both variables with gradients that must be computed and not
    a = Tensor(np.array(5), True)
    b = Tensor(np.array(2))
    r = a * b
    r.back_prop()
    assert not b.grad and not b.c_g_

    # Straightforward chains with several matrices
    a = Tensor(np.array([[3, 2],
                         [5, 7]]), True)
    b = Tensor(np.array([[2],
                         [3]]), True)
    k = Tensor(np.array([3]))
    r = ((k * a) @ b).sum()
    r.back_prop()
    assert r.value[0][0] == 129
    assert a.grad.shape == (2, 2)
    assert b.grad.shape == (2, 1)
    assert a.grad[0][0] == a.grad[1][0] == 6 and a.grad[0][1] == a.grad[1][1] == 9
    assert b.grad[0][0] == 9 + 15 and b.grad[1][0] == 6 + 21

    print('Tensor Chain Rule Test Succeeded')


def log_test():
    """
    Tensor Log Function Test
    """

    a = Tensor(np.array([5]))
    r = log(a)
    assert a.value.shape == r.value.shape == (1, 1)
    assert r.value - math.log(5) < 1e-6
    assert not r.c_g_

    a = Tensor(np.array([5]), True)
    r = log(a)
    r.back_prop()
    assert a.grad.shape == (1, 1)
    assert r.value - math.log(5) < 1e-6
    assert r.c_g_
    assert a.grad[0][0] == 1/5

    a = Tensor(np.array([5, 9]), True)
    r = log(a).sum()
    r.back_prop()
    assert r.value - math.log(45) < 1e-6
    assert a.grad.shape == (2, 1)
    assert a.grad[0][0] == 1/5 and a.grad[1][0] == 1/9

    a = Tensor(np.array([[2, 3], [4, 5]]), True)
    r = log(a).sum()
    r.back_prop()
    assert r.value - math.log(2*3*4*5) < 1e-6
    assert a.grad.shape == (2, 2)
    assert a.grad[0][0] == 1 / 2
    assert a.grad[0][1] == 1 / 3
    assert a.grad[1][0] == 1 / 4
    assert a.grad[1][1] == 1 / 5

    print('Tensor Log Function Tests Succeeded')


def exp_test():
    """
    Tensor Exp Function Test
    """

    a = Tensor(np.array([5]))
    r = exp(a)
    assert a.value.shape == r.value.shape == (1, 1)
    assert r.value - math.exp(5) < 1e-6
    assert not r.c_g_

    a = Tensor(np.array([5]), True)
    r = exp(a)
    r.back_prop()
    assert a.grad.shape == (1, 1)
    assert r.value - math.exp(5) < 1e-6
    assert r.c_g_
    assert a.grad[0][0] == math.exp(5)

    a = Tensor(np.array([5, 9]), True)
    r = exp(a).sum()
    r.back_prop()
    assert r.value - math.exp(5) - math.exp(9) < 1e-6
    assert a.grad.shape == (2, 1)
    assert a.grad[0][0] == math.exp(5)
    assert a.grad[1][0] == math.exp(9)

    a = Tensor(np.array([[2, 3], [4, 5]]), True)
    r = exp(a).sum()
    r.back_prop()
    assert r.value - math.exp(2) - math.exp(3) - math.exp(4) - math.exp(5) < 1e-6
    assert a.grad.shape == (2, 2)
    assert a.grad[0][0] == math.exp(2)
    assert a.grad[0][1] == math.exp(3)
    assert a.grad[1][0] == math.exp(4)
    assert a.grad[1][1] == math.exp(5)

    print('Tensor Exp Function Tests Succeeded')


def sigmoid_test():
    """
    Tensor Sigmoid Function Test
    """

    s = lambda x: 1 / (1 + math.exp(-x))

    a = Tensor(np.array([5]))
    r = sigmoid(a)
    assert a.value.shape == r.value.shape == (1, 1)
    assert r.value - s(5) < 1e-6
    assert not r.c_g_

    a = Tensor(np.array([5]), True)
    r = sigmoid(a)
    r.back_prop()
    assert a.grad.shape == (1, 1)
    assert r.value - s(5) < 1e-6
    assert r.c_g_
    assert a.grad[0][0] - s(5)*(1-s(5)) < 1e-6

    a = Tensor(np.array([5, 9]), True)
    r = sigmoid(a).sum()
    r.back_prop()
    assert r.value - s(5) - s(9) < 1e-6
    assert a.grad.shape == (2, 1)
    assert a.grad[0][0] == s(5)*(1-s(5))
    assert a.grad[1][0] == s(9)*(1-s(9))

    a = Tensor(np.array([[2, 3], [4, 5]]), True)
    r = sigmoid(a).sum()
    r.back_prop()
    assert r.value - s(2) - s(3) - s(4) - s(5) < 1e-6
    assert a.grad.shape == (2, 2)
    assert a.grad[0][0] == s(2)*(1-s(2))
    assert a.grad[0][1] == s(3)*(1-s(3))
    assert a.grad[1][0] == s(4)*(1-s(4))
    assert a.grad[1][1] == s(5)*(1-s(5))

    print('Tensor Sigmoid Function Tests Succeeded')


def ReLU_test():
    """
    Tensor ReLU Function Test
    """

    a = Tensor(np.array([5]))
    b = Tensor(np.array([-5]))
    r = ReLU(a) + ReLU(b)
    assert r.value.shape == (1, 1)
    assert r.value == 5
    assert not r.c_g_

    a = Tensor(np.array([-5, 9]), True)
    r = ReLU(a).sum()
    r.back_prop()
    assert r.value == 9
    assert a.grad.shape == (2, 1)
    assert a.grad[0][0] == 0
    assert a.grad[1][0] == 1

    a = Tensor(np.array([[-2, 3], [4, -5]]), True)
    r = ReLU(a).sum()
    r.back_prop()
    assert r.value == 7
    assert a.grad.shape == (2, 2)
    assert a.grad[0][0] == 0
    assert a.grad[0][1] == 1
    assert a.grad[1][0] == 1
    assert a.grad[1][1] == 0

    print('Tensor ReLU Function Tests Succeeded')


# Tensor methods
initialization_test()
addition_test()
subtraction_test()
multiplication_test()
truediv_test()
matmul_test()
pow_test()
sum_test()
chain_rule_test()

# Helper methods
log_test()
exp_test()
sigmoid_test()
ReLU_test()
