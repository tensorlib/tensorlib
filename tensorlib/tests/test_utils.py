from tensorlib.utils import check_random_state
from nose.tools import assert_raises


def test_random_state():
    check_random_state(None)
    check_random_state(1999)
    assert_raises(ValueError, check_random_state, "invalid")
