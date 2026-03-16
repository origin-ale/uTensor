from copy import copy
import numpy as np
import pytest
import tens_net as tn

@pytest.fixture
def fourlegs():
   return tn.Tensor([[[[1,2],[3,4]],[[5,6],[7,8]]],[[[9,10],[11,12]],[[13,14],[15,16]]]])

@pytest.mark.parametrize(
  "part1, matrixized",
  [
    (
      (2,3), 
      tn.Tensor((np.arange(16)+1).reshape((4,4)))
    ),
    (
      (1,3),
      tn.Tensor([[1,2,5,6], [3,4,7,8], [9,10,13,14], [11,12,15,16]])
    ),
    (
      (0,2),
      tn.Tensor([[1,3,9,11], [2,4,10,12], [5,7,13,15], [6,8,14,16]])
    ),
    (
      (3,),
      tn.Tensor((np.arange(16)+1).reshape((8,2)))
    ),
    (
      (2,),
      tn.Tensor([[1,3], [2,4], [5,7], [6,8], [9,11], [10,12], [13,15], [14,16]])
    ),
    (
      (0,1,3),
      tn.Tensor(np.array([[1,3], [2,4], [5,7], [6,8], [9,11], [10,12], [13,15], [14,16]]).T)
    ),
  ])
def test_matrixize(fourlegs, part1, matrixized):
   t = copy(fourlegs)
   tp, _ = tn.matrixize(t, part1)
   assert tp == matrixized