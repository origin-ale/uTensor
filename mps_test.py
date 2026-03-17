import numpy as np
import tens_net as tn
import mps
import pytest
from copy import copy

@pytest.fixture
def fourlegs():
  return [[[[1,2],[3,4]],[[5,6],[7,8]]],[[[9,10],[11,12]],[[13,14],[15,16]]]]

def test_mps_init():
  state = mps.Mps(N = 3)
  assert len(state) == 3

def test_uniapply(fourlegs):
  state = mps.Mps(N = 3)
  uni = mps.Unitary(fourlegs)
  res = mps.apply_unitary(uni, state[0], state[1])
  assert res.n_legs() == 4
  for (l, d) in zip(range(0,4), (1,1,2,2)):
    assert res.dim_leg(l) == d

def test_mpo_init(fourlegs):
  mpo = mps.MpoNN()
  uni = mps.Unitary(fourlegs)
  mpo.append(uni)
  assert len(mpo) == 1

def test_mpoapply(fourlegs):
  bond_dim = 10
  state = mps.Mps(N=8)
  uni = mps.Unitary(fourlegs)
  mpo = mps.MpoNN()
  for i in range(0,4): mpo.append(copy(uni))
  new_state = mps.apply_mponn(mpo, state, bond_dim)
  assert new_state != state
  assert len(new_state) == 8
  for t in range(0,len(new_state)):
    try: 
      assert new_state[t].dim_leg(1) == 2
      assert new_state[t].dim_leg(2) == new_state[t+1].dim_leg(0)
      if t%2 == 0: assert new_state[t].dim_leg(0) == 1
      else: assert new_state[t].dim_leg(0) in range(2, bond_dim)
    except IndexError: pass