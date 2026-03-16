import numpy as np
import tens_net as tn
import mps
import pytest

@pytest.fixture
def fourlegs():
  return tn.Tensor([[[[1,2],[3,4]],[[5,6],[7,8]]],[[[9,10],[11,12]],[[13,14],[15,16]]]])

def test_init():
  state = mps.Mps(3)

def test_uniapply(fourlegs):
  state = mps.Mps(3)
  mps.apply(fourlegs, state[0], state[1])
  assert False