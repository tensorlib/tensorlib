import numpy as np
from numpy.testing import assert_array_almost_equal
from tensorlib.mathutils import kr, _canonical_kr
from tensorlib.mathutils import matricize


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
    Test implementation of matricize. Data from
    http://www.graphanalysis.org/SIAM-PP08/Dunlavy.pdf
    """
    X = np.arange(1, 9).reshape(2, 2, 2)
    X1 = np.rollaxis(X, 2).reshape(X.shape[2], -1)
    X2 = np.rollaxis(X, 1).reshape(X.shape[1], -1)
    X3 = np.rollaxis(X, 0).reshape(X.shape[0], -1)
    assert_array_almost_equal(X1, matricize(X, 0))
    assert_array_almost_equal(X1, matricize(X, -3))
    assert_array_almost_equal(X2, matricize(X, 1))
    assert_array_almost_equal(X2, matricize(X, -2))
    assert_array_almost_equal(X3, matricize(X, 2))
    assert_array_almost_equal(X3, matricize(X, -1))
