from tensorlib.decomposition import cp
from tensorlib.datasets import load_bread
from nose.tools import assert_raises


def test_cp():
    """
    Test CANDECOMP/PARFAC decomposition. Problem from
    http://issnla2010.ba.cnr.it/DecompositionsI.pdf
    """
    X, meta = load_bread()
    assert_raises(ValueError, cp, X)
    cp(X, 2, init_type="hosvd", random_state=1999)
