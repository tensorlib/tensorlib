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
    matricize(X, 0)
    assert_array_almost_equal(matricize(X, 2), matricize(X, -1))
    assert_array_almost_equal(matricize(X, 1), matricize(X, -2))
