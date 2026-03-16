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

def test_conservation(threelegs, fourlegs):
  t1 = threelegs
  t2 = fourlegs
  lhs1, rhs1 = tn.svd(t1)
  lhs2, rhs2 = tn.svd(t2)

  assert tn.contract(lhs1, rhs1, lhs1.n_legs()-1, 0).elements == pytest.approx(threelegs.elements)
  assert tn.contract(lhs2, rhs2, lhs1.n_legs()-1, 0).elements == pytest.approx(fourlegs.elements)

def test_nlegs(threelegs, fourlegs):
  t1 = threelegs
  t2 = fourlegs
  lhs1, rhs1 = tn.svd(t1)
  lhs2, rhs2 = tn.svd(t2)
  lhs3, rhs3 = tn.svd(t2, rhs_legnum = 3)
  assert lhs1.n_legs() == 3
  assert rhs1.n_legs() == 2
  assert lhs2.n_legs() == 3
  assert rhs2.n_legs() == 3
  assert lhs3.n_legs() == 2
  assert rhs3.n_legs() == 4

def test_dimensions(threelegs, fourlegs):
  t1 = threelegs
  t2 = fourlegs
  lhs1, rhs1 = tn.svd(t1, bond_dim = 1)
  lhs2, rhs2 = tn.svd(t2, bond_dim = 1)
  assert lhs1.dim_leg(lhs1.n_legs()-1) == 1
  assert rhs1.dim_leg(0) == 1
  assert lhs2.dim_leg(lhs2.n_legs()-1) == 1
  assert rhs2.dim_leg(0) == 1