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

@pytest.fixture
def threelegs():
   return tn.Tensor(
      np.array(
          [[[1,2,3,4],[5,6,7,8]],
           [[9,10,11,12],[13,14,15,16]],
           [[17,18,19,20],[21,22,23,24]]
          ]
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

@pytest.mark.parametrize(
    "start_tensor,bundle_legs,bundled_tensor",
    [
      (tn.Tensor(
          [[[1,2,3,4],[5,6,7,8]],
           [[9,10,11,12],[13,14,15,16]],
           [[17,18,19,20],[21,22,23,24]]
          ]
        ),
        (1,2), 
        tn.Tensor(
          [[1,2,3,4,5,6,7,8],
           [9,10,11,12,13,14,15,16],
           [17,18,19,20,21,22,23,24]
          ]
        )
      ),
      (tn.Tensor(
          [[[1,2,3,4],[5,6,7,8]],
           [[9,10,11,12],[13,14,15,16]],
           [[17,18,19,20],[21,22,23,24]]
          ]
        ),
        (0,1), 
        tn.Tensor(
          [[1,2,3,4],
           [5,6,7,8],
           [9,10,11,12],
           [13,14,15,16],
           [17,18,19,20],
           [21,22,23,24]
          ]
        )
      ),
      (tn.Tensor(
          [[[1,2,3,4],[5,6,7,8]],
           [[9,10,11,12],[13,14,15,16]],
           [[17,18,19,20],[21,22,23,24]]
          ]
        ),
        (0,2), 
        tn.Tensor(
          [[1,5], [2,6], [3,7], [4,8], [9,13], [10,14], 
           [11,15], [12,16], [17,21], [18,22], [19,23], [20,24]
          ]
        )
      ),
    ]
)
class TestBundle:
   def test_bundle(self, start_tensor, bundle_legs, bundled_tensor):
      t = copy(start_tensor)
      t.bundle_legs(*bundle_legs)
      t == start_tensor
      assert not t == start_tensor
      assert t == bundled_tensor

        # tn.Tensor(
        #   [[1,5,9,13,17,21],
        #    [2,6,10,14,18,22],
        #    [3,7,11,15,19,23],
        #    [4,8,12,16,20,24]
        #   ]
        # )