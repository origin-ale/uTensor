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
  
  def svd_sweep(self, bond_dim = None):
    last_point = -1
    new_factors = []
    idn = Unitary([[[[1,0]], [[0,1]]]])
    for t in self:

      new_point = min(t.points)
      # print(f"{new_point=} {last_point=}")
      if new_point <= last_point: raise ValueError("Unitaries must be sorted by point of application")
      elif new_point > last_point+1: 
        for i in range(last_point, new_point): 
          new_factors.append(AppliedUnitary(idn, (i,)))
          print(f"Put identity in position {i}")

      if len(t.points) == 1:
        new_factors.append(t)
      elif len(t.points) == 2:
        # print(f"{t.uni.dim_legs()=}")
        t0, t1 = tn.svd(t.uni, bond_dim, absorb_sv=1)
        # print(f"{t0.dim_legs()=} {t1.dim_legs()=}")
        if t0.n_legs() == 3: t0.add_leg(0)
        if t1.n_legs() == 3: t1.add_leg(3)
        # print(f"{t0.dim_legs()=} {t1.dim_legs()=}")
        t0.move_leg(1,3)
        t1.move_leg(1,3)
        new_factors.append(AppliedUnitary(t0, (t.points[0],)))
        new_factors.append(AppliedUnitary(t1, (t.points[1],)))
      else: raise ValueError("SVD for unitaries applied to more than 2 factors not implemented yet")
      last_point = max(t.points)
      # print(f"{len(new_factors)=}")
    return Mpo(new_factors)

def apply_bond_unitary(U, s1, s2):
  c1 = tn.contract(U, s1, 0, 1)
  c2 = tn.contract(c1, s2, 0, 1)
  c3 = tn.trace(c2, 3, 4)
  c3.move_leg(2,0)
  c3.move_leg(3,0)
  return c3

def apply_site_unitary(op: AppliedUnitary, state: Mps):
  mps_fac = state[op.points[0]]
  print(f"{op.uni.dim_legs()} -- {mps_fac.dim_legs()}")
  c1 = tn.contract(op.uni, mps_fac, 1, 1)
  print(f"Before bundling: {c1.dim_legs()}")
  c1.bundle_legs(0,4)
  c1.bundle_legs(1,3)
  c1.move_leg(1,2)
  print(f"After bundling: {c1.dim_legs()}")
  print("-----")
  return c1

def apply_site_mpo(op: Mpo, state: Mps):
  new_factors = [apply_site_unitary(t, state) for t in op]
  return Mps(new_factors)

def apply_bond_mpo(op: Mpo, state: Mps, bonddim_ = None):
  new_factors = []
  for i in range(0, len(op)):
    undecomposed = apply_bond_unitary(op[i].uni, state[op[i].points[0]], state[op[i].points[1]])
    svd_i = tn.svd(undecomposed, bond_dim = bonddim_, absorb_sv = 1)
    new_factors.extend(copy(svd_i))
  if len(state) - len(new_factors) == 2:
    new_factors.insert(0, copy(state[0]))
    new_factors.append(copy(state[-1]))
  elif len(state) - len(new_factors) == 1:
    if op[0].points[0] == 0:
      new_factors.append(copy(state[-1]))
    elif op[0].points[0] == 1:
      new_factors.insert(0, copy(state[0]))
  assert len(new_factors) == len(state)
  return Mps(new_factors)