import numpy as np
from tensorlib.decomposition import cp
from tensorlib.decomposition.decomposition import _cp3
from tensorlib.decomposition import tucker
from tensorlib.decomposition.decomposition import _tucker3
from tensorlib.datasets import load_bread
from numpy.testing import assert_almost_equal
from nose.tools import assert_raises


def test_generated_cp():
    """
    Test CANDECOMP/PARFAC decomposition. Problem from
    http://issnla2010.ba.cnr.it/DecompositionsI.pdf
    """
    rs = np.random.RandomState(1999)
    X = .7 * rs.rand(2, 4, 3) + .25 * rs.rand(2, 4, 3)
    assert_raises(ValueError, cp, X)
    U1 = cp(X, 2, init_type="hosvd")
    U2 = _cp3(X, 2, tol=1E-4, max_iter=500, init_type="hosvd")
    for n, i in enumerate(U1):
        assert_almost_equal(U1[n], U2[n])


def test_bread_cp():
    """
    Test CANDECOMP/PARFAC decomposition using bread dataset.
    """
    X, meta = load_bread()
    assert_raises(ValueError, cp, X)
    U1 = cp(X, 2, init_type="hosvd")
    U2 = _cp3(X, 2, tol=1E-4, max_iter=500, init_type="hosvd")
    for n, i in enumerate(U1):
        assert_almost_equal(U1[n], U2[n])


def test_generated_tucker():
    """
    Test CANDECOMP/PARFAC decomposition. Problem from
    http://issnla2010.ba.cnr.it/DecompositionsI.pdf
    """
    rs = np.random.RandomState(1999)
    X = .7 * rs.rand(2, 4, 3) + .25 * rs.rand(2, 4, 3)
    assert_raises(ValueError, cp, X)
    U1 = tucker(X, 2, init_type="hosvd")
    U2 = _tucker3(X, 2, tol=1E-4, max_iter=500, init_type="hosvd")
    for n, i in enumerate(U1):
        assert_almost_equal(U1[n], U2[n])
