import numpy as np
import itertools as it
from copy import copy

class LinkingError: Exception

class Tensor:
  def __init__(self, ndarr: np.ndarray | list):
    self.elements = np.array(ndarr)

  def __eq__(self, value):
    if self.elements.shape == value.elements.shape:
      return np.all(self.elements == value.elements)
    else:
      return False
    
  def __repr__(self):
    return f"Tensor({self.elements.__repr__()})"

  def n_legs(self):
    return self.elements.ndim
  
  def dim_leg(self, leg: int):
    return np.size(self.elements, axis = leg)
  
  def move_leg(self, source, target):
    self.elements = np.moveaxis(self.elements, source, target)
         
  def bundle_legs(self, leg1, leg2):
    """Bundle two legs. 
    Legs are bundled together at the lowest-numbered leg."""
    legs = (leg1, leg2)
    el = self.elements
    bundled_leg = min(legs)
    elm = np.moveaxis(el, legs, (0,1))
    elmp = np.concat(np.unstack(elm, axis=0), axis=0)
    self.elements = np.moveaxis(elmp, 0, bundled_leg)

  def unbundle_leg(self, leg, leg_dims):
    """Unbundle one leg into multiple legs with dimensions specified via tuple.
    Legs appear in the specified order starting at the current position."""
    leg_n = len(leg_dims)
    el = self.elements
    elm = np.moveaxis(el, leg, 0)
    split_tens = elm
    split_dim = np.prod(leg_dims)
    if split_dim != self.dim_leg(leg):
      raise ValueError(f"Cannot unbundle leg of dimension {self.dim_leg(leg)} into legs of dimensions {leg_dims}.")
    curr_leg = leg_n-1
    while curr_leg > 0:
      split_tens = np.stack(np.split(split_tens, split_dim/leg_dims[curr_leg], axis=0), axis = 0)
      split_dim /= 2
      curr_leg -= 1
    elmp = split_tens
    self.elements = np.moveaxis(elmp,
                                tuple(range(leg_n)),
                                tuple(range(leg, leg + leg_n)))
  
def matrixize(op_o, leg):
  op = copy(op_o)
  uncontracted_legs = list(range(op.n_legs()))
  uncontracted_legs.pop(leg)
  op.move_leg(leg, -1)
  uncontracted_dims = []
  uncontracted_dims.append(op.dim_leg(0))
  while op.n_legs() > 2:
    uncontracted_dims.append(op.dim_leg(1))
    op.bundle_legs(0,1)
  
  return op, uncontracted_dims
  

def contract(op1, op2, leg1, leg2):
  op = []
  uncontracted_dims = []
  for o,l in ((op1, leg1), (op2, leg2)):
    temp_op, temp_ud = matrixize(o, l)
    op.append(temp_op)
    uncontracted_dims.append(temp_ud)
  result = Tensor(op[0].elements @ op[1].elements.T)
  for i in (1,0):
    result.unbundle_leg(i, uncontracted_dims[i])
  return result

def svd(op, bond_dim = None, absorb_sv = 'R' | 'L'):
  pass