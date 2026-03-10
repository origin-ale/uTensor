import numpy as np
import pytest
import tens_net as tn

@pytest.mark.parametrize(
    "ndarr",
    [
        np.array([1, 2, 3, 4]),
        np.array([[1, 2], [3, 4]]),
        np.array([[1,2,3],[4,5,6]]),
        np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]],[[13,14,15],[16,17,18]]])
    ],
)

def test_equals(ndarr):
    t = tn.Tensor(ndarr)
    assert np.array_equal(t.elements, ndarr)