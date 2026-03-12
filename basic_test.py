from copy import copy
import numpy as np
import pytest
import tens_net as tn

@pytest.mark.parametrize(
  "ndarr",
  [
    np.array([1, 2, 3, 4]),
    np.array([[1, 2], [3, 4]]),
    np.array([[1,2,3],[4,5,6]]),
    np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]],[[13,14,15],[16,17,18]]]),
    [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]],[[13,14,15],[16,17,18]]]
  ],
)
class TestSingleTensors:
  def test_elements(self, ndarr):
    t = tn.Tensor(ndarr)
    assert np.array_equal(t.elements, ndarr)

  def test_legs(self, ndarr):
    t = tn.Tensor(ndarr)
    assert t.n_legs() == np.ndim(ndarr) # ndarr is a list, but np.ndim 
                                        # converts it to an array

@pytest.mark.parametrize(
  "start,move,expected",
  [
    (np.arange(4), (0,0), np.arange(4)),
    (np.arange(4).reshape(2,2), (0,1), np.arange(4).reshape(2,2).T),
    (np.arange(6).reshape(2,3), (0,1), np.arange(6).reshape(2,3).T),
    ((np.arange(18)+1).reshape(3,2,3), (0,2), 
     [[[1,7,13],[2,8,14],[3,9,15]], [[4,10,16],[5,11,17],[6,12,18]]]),
  ],
)
def test_leg_moving(start, move, expected):
  t = tn.Tensor(start)
  t.move_leg(*move)
  assert t == tn.Tensor(expected)