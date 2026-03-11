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
class TestSingleTensors:
  def test_elements(self, ndarr):
      t = tn.Tensor(ndarr)
      assert np.array_equal(t.elements, ndarr)

  def test_legs(self, ndarr):
      t = tn.Tensor(ndarr)
      assert t.n_legs() == np.ndim(ndarr) # ndarr is a list, but np.ndim 
                                          # converts it to an array

@pytest.fixture
def threelegs():
   return tn.Tensor(
      np.array(
         [[[1,2,3,4],[5,6,7,8]],[[9,10,11,12],[13,14,15,16]],[[17,18,19,20],[21,22,23,24]]]
         )
      )

class TestExplicit:
  def test_nlegs(self, threelegs):
    assert threelegs.n_legs() == 3

  def test_legdims(self, threelegs):
    assert threelegs.dim_leg(0) == 3
    assert threelegs.dim_leg(1) == 2
    assert threelegs.dim_leg(2) == 4
  
  def test_outrange(self, threelegs):
     with pytest.raises(IndexError):
        threelegs.dim_leg(3)