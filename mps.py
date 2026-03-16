import numpy as np
import tens_net as tn

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

class Mps:
  def __init__(self, N: int, init_factors = None):
    if init_factors is not None:
      raise Exception("Explicitly initializing MPS is not supported yet")
    else:
      self.factors = [MpsElement([[[1,],[0,]]]) for i in range(0,N)]

  def __getitem__(self, key):
    return self.factors[key]

def apply(U, s1, s2):
  print(f"U: {U.elements.shape}\ns1:{s1.elements.shape}")
  c1 = tn.contract(U, s1, 0, 1)
  print(f"c1: {c1.elements.shape}\ns2:{s2.elements.shape}")
  c2 = tn.contract(c1, s2, 0, 1)
  print(f"c2: {c2.elements.shape}")
  c3 = tn.trace(c2, 3, 4)
  print(f"c3: {c3.elements.shape}")
  c3.move_leg(0,2)
  c3.move_leg(0,3)
  print(c3)
  return c3