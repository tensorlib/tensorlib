import numpy as np
from numpy.testing import assert_array_almost_equal, assert_raises
from tensorlib.mathutils import kr, _canonical_kr
from tensorlib.mathutils import matricize, unmatricize, tmult


def test_kr():
    """
    Test correctness of Khatri-Rao product. Data from
    http://gmao.gsfc.nasa.gov/events/adjoint_workshop-8/present/Friday/Dance.pdf
    """
    B = np.arange(1, 5).reshape(2, 2)
    C = np.arange(5, 11).reshape(3, 2)
    A = kr(B, C)

    expected_result = np.array([[5, 12],
                                [7, 16],
                                [9, 20],
                                [15, 24],
                                [21, 32],
                                [27, 40]])

    assert_array_almost_equal(A, expected_result)


def test_canonical_kr():
    """
    Tests the equivalence of kr implementation and canonical np.kron form.
    """
    B = np.arange(1, 5).reshape(2, 2)
    C = np.arange(5, 11).reshape(3, 2)
    A = kr(B, C)
    A_canon = _canonical_kr(B, C)

    assert_array_almost_equal(A, A_canon)


def test_matricize():
    """
    Test implementation of matricize.
    """
    X = np.arange(1, 9).reshape(2, 2, 2)
    matricize(X, 0)
    assert_array_almost_equal(matricize(X, 2), matricize(X, -1))
    assert_array_almost_equal(matricize(X, 1), matricize(X, -2))


def test_unmatricize():
    """
    Test implementation of unmatricize.
    """
    X = np.arange(2 * 3 * 4 * 5).reshape(3, 2, 4, 5)
    for i in range(X.ndim):
        X2 = matricize(X, i)
        assert_array_almost_equal(unmatricize(X2, i, X.shape), X)

    X2 = matricize(X, -2)
    assert_array_almost_equal(unmatricize(X2, -2, X.shape), X)

    X = np.arange(1, 9).reshape(2, 2, 2)
    for i in range(X.ndim):
        X2 = matricize(X, i)
        assert_array_almost_equal(unmatricize(X2, i, X.shape), X)


def test_tmult():
    X1 = np.arange(1, 25).reshape(2, 3, 4)
    X2 = np.arange(9).reshape(3, 3)
    tmult(X1, X2, 1)
    assert_raises(ValueError, tmult, X1, X2, 0)
