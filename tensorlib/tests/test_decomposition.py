import numpy as np
from numpy.testing import assert_array_almost_equal
from tensorlib.decomposition import cp
from nose.tools import assert_raises


def test_cp():
    """
    Test CANDECOMP/PARFAC decomposition. Problem from
    http://issnla2010.ba.cnr.it/DecompositionsI.pdf
    """
    rs = np.random.RandomState(999)
    X = rs.randn(3, 5, 4)
    assert_raises(ValueError, cp, X)
    A1, B1, C1 = cp(X, 2, random_state=1999)
    A2, B2, C2 = cp(X, 2, random_state=1999, force_general=True)
    assert_array_almost_equal(A1, A2)
    assert_array_almost_equal(B1, B2)
    assert_array_almost_equal(C1, C2)
