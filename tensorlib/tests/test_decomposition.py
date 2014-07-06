import numpy as np
from tensorlib.decomposition import cp
from nose.tools import assert_raises


def test_cp():
    """
    Test CANDECOMP/PARFAC decomposition. Problem from
    http://issnla2010.ba.cnr.it/DecompositionsI.pdf
    """
    rs = np.random.RandomState(999)
    X = rs.randn(5, 6, 7)
    assert_raises(ValueError, cp, X)
    A, B, C = cp(X, 2, random_state=1999)
