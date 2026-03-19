import numpy as np
import tens_net as tn
import mps
import pytest
from copy import copy

@pytest.fixture
def fourlegs():
  return [[[[1,2],[3,4]],[[5,6],[7,8]]],[[[9,10],[11,12]],[[13,14],[15,16]]]]

@pytest.fixture
def swap():
  return [[[[1,0],[0,0]],[[0,0],[1,0]]], [[[0,1],[0,0]],[[0,0],[0,1]]]]

def test_mps_init():
  state = mps.Mps(N = 3)
  assert len(state) == 3

def test_uniapply(fourlegs):
  state = mps.Mps(N = 3)
  uni = mps.Unitary(fourlegs)
  res = mps.apply_bond_unitary(uni, state[0], state[1])
  assert res.n_legs() == 4
  for (l, d) in zip(range(0,4), (1,1,2,2)):
    assert res.dim_leg(l) == d

def test_mpo_init(fourlegs):
  mpo = mps.Mpo()
  uni = mps.Unitary(fourlegs)
  auni = mps.AppliedUnitary(uni, (0,1))
  mpo.append(auni)
  assert len(mpo) == 1

@pytest.mark.parametrize(
    "N",
    [2, 8, 3, 7]
)
def test_mpoapply(fourlegs, N):
  bond_dim = 10
  state = mps.Mps(N=N)
  uni = mps.Unitary(fourlegs)
  mpo = mps.Mpo()
  for i in range(0,N//2):
    auni = mps.AppliedUnitary(copy(uni), (2*i, 2*i+1))
    mpo.append(auni)
  # print(len(mpo), len(state))
  idn = mps.Unitary([[[[1,0]], [[0,1]]]])
  mpo = mpo.svd_sweep()
  if(len(state)-len(mpo)) == 1:
    mpo.append(mps.AppliedUnitary(copy(idn), (len(state)-1,))) 
  # print(len(mpo), len(state))
  new_state = mps.apply_site_mpo(mpo, state)
  assert new_state != state
  assert len(new_state) == N
  assert new_state[0].dim_leg(0) == 1
  assert new_state[N-1].dim_leg(2) == 1
  for t in range(0,len(new_state)):
    try: 
      assert new_state[t].dim_leg(1) == 2
      assert new_state[t].dim_leg(2) == new_state[t+1].dim_leg(0)
      if t%2 == 0: assert new_state[t].dim_leg(0) == 1
      else: assert new_state[t].dim_leg(0) in range(2, bond_dim)
    except IndexError: pass

@pytest.mark.skip
def test_swap(swap):
  comp0 = mps.MpsElement([[[1],[0]]])
  comp1 = mps.MpsElement([[[1],[0]]])
  comp2 = mps.MpsElement([[[0],[1]]])
  comp3 = mps.MpsElement([[[0],[1]]])
  state = mps.Mps(init_factors=[comp0, comp1, comp2, comp3])
  mpo = mps.Mpo([mps.AppliedUnitary(mps.Unitary(swap), (1,2))]) #do it on the center
  finstate = mps.apply_bond_mpo(mpo, state)
  assert finstate[0] == comp0
  assert finstate[1] == comp2
  assert finstate[2] == comp1
  assert finstate[3] == comp3
