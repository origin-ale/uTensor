from copy import copy
import numpy as np
import pytest
import tens_net as tn

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

@pytest.fixture
def fourlegs():
  return tn.Tensor([[[[1,2],[3,4]],[[5,6],[7,8]]],[[[9,10],[11,12]],[[13,14],[15,16]]]])

@pytest.mark.parametrize(
  "legs_to_bundle, legs_to_unbundle, legs_to_move",
  [
    (
      ((1, 2),), 
      ((1, (2, 4)),), 
      ()
    ),
    (
      ((1, 2), (0, 1)), 
      ((0, (3, 8)), (1, (2, 4))), 
      ()
    ),
    (
      ((2, 0),), 
      ((0, (4,3)),), 
      ((0,2),)
    ),
    (
      ((2, 0), (0, 1)), 
      ((0, (12,2)), (0, (4,3))), 
      ((0,2),)
    ),
  ],
)
def test_three(threelegs, legs_to_bundle, legs_to_unbundle, legs_to_move):
  t = copy(threelegs)
  for legs in legs_to_bundle:
    t.bundle_legs(*legs)
  for leg_to_unbundle, unbundle_dims in legs_to_unbundle:
    t.unbundle_leg(leg_to_unbundle, unbundle_dims)
  for source, target in legs_to_move:
    t.move_leg(source, target)
  assert t == threelegs
   
@pytest.mark.parametrize(
  "legs_to_bundle, legs_to_unbundle, legs_to_move",
  [
    (
      ((1, 2),), 
      ((1, (2, 2)),), 
      ()
    ),
    (
      ((1, 2), (0, 1)), 
      ((0, (2, 4)), (1, (2, 2))), 
      ()
    ),
    (
      ((1, 2), (0, 1), (0,1)), 
      ((0, (8, 2)), (0, (4, 2)), (0, (2,2))), 
      ()
    ),
    (
      ((0, 2),), 
      ((0, (2,2)),), 
      ((1,2),)
    ),
  ],
)
def test_four(fourlegs, legs_to_bundle, legs_to_unbundle, legs_to_move):
  t = copy(fourlegs)
  for legs in legs_to_bundle:
    t.bundle_legs(*legs)
  for leg_to_unbundle, unbundle_dims in legs_to_unbundle:
    t.unbundle_leg(leg_to_unbundle, unbundle_dims)
  for source, target in legs_to_move:
    t.move_leg(source, target)
  assert t == fourlegs