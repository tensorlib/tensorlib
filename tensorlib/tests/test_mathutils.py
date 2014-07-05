import numpy as np
from numpy.testing import assert_array_almost_equal
from tensorlib.mathutils import kr, _kr_einsum


# def test_kr():
#     """Test correctness of Khatri-Rao product using wikipedia example."""
#     B = np.arange(1, 10, 1).reshape(3, 3)
#     C = np.arange(1, 10, 1).reshape(3, 3).T
#     right_answer = np.array([[1, 2, 12, 21],
#                              [4, 5, 24, 42],
#                              [14, 16, 45, 72],
#                              [21, 24, 54, 81]])
#     A = kr(B, C)
#     print(A)
#     assert_array_almost_equal(A, right_answer)


def test_kr_einsum_kr_equiv():
    """Tests the equivalence of kr and _kr_einsum"""

    rng = np.random.RandomState(42)
    m, n, p = 4, 5, 6

    B = rng.randn(n, p)
    C = rng.randn(m, p)

    A = kr(B, C)
    A_einsum = _kr_einsum(B, C)

    assert_array_almost_equal(A, A_einsum)
