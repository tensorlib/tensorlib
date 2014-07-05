import numpy as np
from numpy.testing import assert_array_almost_equal
from tensorlib.mathutils import kr, _kr_einsum, _kr_npdot


def test_kr():
    """Test correctness of Khatri-Rao product"""

    B = np.eye(3)
    C = np.arange(12).reshape(3, 4).T

    A = kr(B, C)

    expected_result = np.array([[0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 4, 5, 6, 7, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11]]).T

    assert_array_almost_equal(A, expected_result)


def test_kr_einsum_kr_equiv():
    """Tests the equivalence of kr and _kr_einsum"""

    rng = np.random.RandomState(42)
    m, n, p = 4, 5, 6

    B = rng.randn(n, p)
    C = rng.randn(m, p)

    A = kr(B, C)
    A_einsum = _kr_einsum(B, C)

    assert_array_almost_equal(A, A_einsum)
