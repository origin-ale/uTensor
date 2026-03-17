import numpy as np
import tens_net as tn
import dataclasses
from copy import copy

class MpsElement(tn.Tensor):
  def __init__(self, ndarr: np.ndarray | list):
    buf = np.array(ndarr)
    try: 
      assert buf.ndim == 3
    except AssertionError: 
      raise ValueError("Wrong number of dimensions: MPS element must have 3 legs")
    self.elements = buf

class Unitary(tn.Tensor):
  def __init__(self, ndarr: np.ndarray | list):
    buf = np.array(ndarr)
    try: 
      assert buf.ndim == 4
    except AssertionError: 
      raise ValueError("Wrong number of dimensions: unitary must have 4 legs")
    self.elements = buf

class Mps(list):
  def __init__(self, init_factors = None, N = None | int):
    if init_factors is not None:
      super().__init__(init_factors)
    elif N is not None:
      list.__init__(self, [MpsElement([[[1,],[0,]]]) for i in range(0,N)])
    else: raise ValueError("At least one of init_factors and N must be provided")

@dataclasses.dataclass
class AppliedUnitary:
  uni: Unitary
  points: tuple

class Mpo(list):
  def __setitem__(self, key, value):
    try: assert type(value) == AppliedUnitary
    except AssertionError: raise TypeError("Only AppliedUnitary instances can be factors of an MPO")
    super().__setitem__(key, value)

def apply_unitary(U, s1, s2):
  c1 = tn.contract(U, s1, 0, 1)
  c2 = tn.contract(c1, s2, 0, 1)
  c3 = tn.trace(c2, 3, 4)
  c3.move_leg(2,0)
  c3.move_leg(3,0)
  return c3

def apply_mponn(op: Mpo, state: Mps, bonddim_ = 10):
  new_factors = []
  for i in range(0, len(op)):
    undecomposed = apply_unitary(op[i].uni, state[op[i].points[0]], state[op[i].points[1]])
    svd_i = tn.svd(undecomposed, bond_dim = bonddim_, absorb_sv = 1)
    new_factors.extend(copy(svd_i))
  if len(state) - len(new_factors) == 2:
    new_factors.insert(0, copy(state[0]))
    new_factors.append(copy(state[-1]))
  assert len(new_factors) == len(state)
  return Mps(new_factors)