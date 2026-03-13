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

@pytest.mark.parametrize(
    "legs_to_contract, einsum_indices",
    [
      (
        (0, 2), 
        'ijk, lmi'
      ),
      (
        (0, 0), 
        'ijk, ilm'
      ),
      (
        (2, 0), 
        'ijk, klm'
      ),
    ],
)
class TestContractionsThree:
  def test_same(self, threelegs, legs_to_contract, einsum_indices):
    t = copy(threelegs)
    contraction = tn.contract(t, t, *legs_to_contract)
    assert contraction == tn.Tensor(np.einsum(einsum_indices, t.elements, t.elements))

  def test_diff(self, threelegs, legs_to_contract, einsum_indices):
    t1 = copy(threelegs)
    t2 = tn.Tensor(threelegs.elements.swapaxes(0,2) + 20)
    contraction = tn.contract(t1, t2, *legs_to_contract)
    assert contraction == tn.Tensor(np.einsum(einsum_indices, t1.elements, t2.elements))

@pytest.mark.parametrize(
    "legs_to_contract, einsum_indices",
    [
      (
        (0, 2), 
        'ijkl, mnip'
      ),
      (
        (0, 0), 
        'ijkl, imnp'
      ),
      (
        (2, 0), 
        'ijkl, kmnp'
      ),
      (
        (1, 3), 
        'ijkl, mnpj'
      ),
      (
        (2, 3), 
        'ijkl, mnpk'
      ),
      (
        (3, 0), 
        'ijkl, lmnp'
      ),
    ],
)
class TestContractionsFour:
  def test_same(self, fourlegs, legs_to_contract, einsum_indices):
    t = copy(fourlegs)
    contraction = tn.contract(t, t, *legs_to_contract)
    assert contraction == tn.Tensor(np.einsum(einsum_indices, t.elements, t.elements))
  
  def test_diff(self, fourlegs, legs_to_contract, einsum_indices):
    t1 = copy(fourlegs)
    t2 = tn.Tensor(np.moveaxis(fourlegs.elements, 1,3) + 20)
    contraction = tn.contract(t1, t2, *legs_to_contract)
    assert contraction == tn.Tensor(np.einsum(einsum_indices, t1.elements, t2.elements))

@pytest.mark.parametrize(
    "legs_to_contract, einsum_indices",
    [
      (
        (1, 0), 
        'ijk, jlmn'
      ),
      (
        (1, 1), 
        'ijk, ljmn'
      ),
      (
        (1, 2), 
        'ijk, lmjn'
      ),
      (
        (1, 3), 
        'ijk, lmnj'
      ),
    ],
)
def test_threefour(threelegs, fourlegs, legs_to_contract, einsum_indices):
  t1 = copy(threelegs)
  t2 = copy(fourlegs)
  contraction = tn.contract(t1, t2, *legs_to_contract)
  assert contraction == tn.Tensor(np.einsum(einsum_indices, t1.elements, t2.elements))