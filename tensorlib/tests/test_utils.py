import numpy as np
from tensorlib.utils import check_random_state
from tensorlib.utils import check_tensor
from tensorlib.utils import downloader
from nose.tools import assert_raises
from nose.plugins.skip import SkipTest


def test_check_tensor():
    rs = np.random.RandomState(1999)
    X = rs.randn(2, 3)
    assert_raises(ValueError, check_tensor, X)
    X = rs.randn(2, 3, 4)
    check_tensor(X)


def test_random_state():
    check_random_state(None)
    check_random_state(1999)
    assert_raises(ValueError, check_random_state, "invalid")


def test_downloader():
    raise SkipTest
    downloader("https://dl.dropboxusercontent.com/u/15378192/Sensory_Bread.zip",
               "Sensory_Bread.zip")
