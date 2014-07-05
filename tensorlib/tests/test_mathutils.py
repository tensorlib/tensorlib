import numpy as np
from numpy.testing import assert_almost_equal
from tensorlib.mathutils import kr


def test_kr():
    """Test correctness of Khatari-Rao product using wikipedia example."""
    B = np.arange(1, 10, 1).reshape(3, 3)
    C = np.arange(1, 10, 1).reshape(3, 3).T
    right_answer = np.array([[1, 2, 12, 21],
                             [4, 5, 24, 42],
                             [14, 16, 45, 72],
                             [21, 24, 54, 81]])
    A = kr(B, C)
    print(A)
    assert_almost_equal(A, right_answer)
