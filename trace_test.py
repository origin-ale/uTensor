from copy import copy
import numpy as np
import pytest
import tens_net as tn

@pytest.fixture
def threelegs():
  return tn.Tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]],[[13,14,15],[16,17,18]]])

@pytest.fixture
def fourlegs():
  return tn.Tensor([[[[1,2],[3,4]],[[5,6],[7,8]]],[[[9,10],[11,12]],[[13,14],[15,16]]]])

def test_three(threelegs):
  t = threelegs
  assert tn.trace(t, 0,2) == tn.Tensor([24,33])

@pytest.mark.parametrize(
    "trace_legs, result",
    [
      (
        (0,1),
        [[14,16],[18,20]]
      ),
      (
        (2,3),
        [[5,13],[21,29]]
      ),
      (
        (0,2),
        [[12,14],[20,22]]
      ),
    ]
)
def test_four(fourlegs, trace_legs, result):
  t = fourlegs
  assert tn.trace(t, *trace_legs) == tn.Tensor(result)