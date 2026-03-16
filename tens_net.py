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
      # print(f"{split_dim=}, {leg_dims=}, {curr_leg=}")
      split_tens = np.stack(np.split(split_tens, split_dim/leg_dims[curr_leg], axis=0), axis = 0)
      split_dim /= leg_dims[curr_leg]
      curr_leg -= 1
    # print(16*'-')
    elmp = split_tens
    self.elements = np.moveaxis(elmp,
                                tuple(range(leg_n)),
                                tuple(range(leg, leg + leg_n)))
  
def matrixize(op_o, legs):
  op = copy(op_o)
  leg_partition = []
  leg_partition.append(list(range(op.n_legs())))
  leg_partition.append([])
  # print(f"Starting from {leg_partition}")
  offset = 0
  for leg in legs:
    leg_partition[0].remove(leg)
    leg_partition[1].append(leg)
    op.move_leg(leg-offset, op.n_legs()-1)
    # print(f"After moving original leg {leg} (ie. leg {leg-offset}): {leg_partition}")
    # print(op.elements)
    offset += 1

  partition_dims = ([],[])

  # print("Bundling lower legs...")
  partition_dims[0].append(op.dim_leg(0))
  # curr = 0
  while op.n_legs() > len(legs)+1:
    partition_dims[0].append(op.dim_leg(1))
    op.bundle_legs(0,1)
    # print(f"Bundled original legs {curr} and {curr+1}")
    # print(op.elements)
    # curr += 1

  # print("Bundling upper legs...")
  partition_dims[1].append(op.dim_leg(op.n_legs()-1))
  # curr = op.n_legs()
  while op.n_legs() > 2:
    partition_dims[1].append(op.dim_leg(op.n_legs()-2))
    op.bundle_legs(op.n_legs()-2,op.n_legs()-1) # bundle_legs cannot handle negative indices
    # print(f"Bundled original legs {curr-1} and {curr}")
    # print(op.elements)
    # curr -= 1
  # print("Done")
  
  return op, partition_dims
  
def trace(op, leg1, leg2):
  return Tensor(np.trace(op.elements, axis1=leg1, axis2=leg2))

def contract(op1, op2, leg1, leg2):
  op = []
  uncontracted_dims = []
  for o,l in ((op1, leg1), (op2, leg2)):
    temp_op, (temp_ud, _) = matrixize(o, (l,))
    op.append(temp_op)
    uncontracted_dims.append(temp_ud)
    # print(f"{uncontracted_dims=}")
  result = Tensor(op[0].elements @ op[1].elements.T)
  for i in (1,0):
    result.unbundle_leg(i, uncontracted_dims[i])
  # print(16*'-')
  return result

def svd(op, bond_dim = None, *, rhs_legnum = None, absorb_sv = 0):
  if rhs_legnum is None:
    rhs_legnum = op.n_legs() // 2
  
  rhs_legs = range(op.n_legs()-rhs_legnum, op.n_legs())
  matrix, partition_dims = matrixize(op, rhs_legs)
  U, lmb, Vh = np.linalg.svd(matrix.elements, 
                             full_matrices=False, 
                             compute_uv = True)
  if bond_dim is not None:
    lmb = lmb[0:bond_dim]
    U = U[:, 0:bond_dim]
    Vh = Vh[0:bond_dim, :]
  
  if absorb_sv == 0:
    U = U @ np.diag(lmb)
  else:
    Vh = np.diag(lmb) @ Vh
  
  lhs = Tensor(U)
  lhs.unbundle_leg(0, partition_dims[0])

  rhs = Tensor(Vh)
  rhs.unbundle_leg(1, partition_dims[1])

  # for tens in (lhs, rhs):
  #   for leg in range(0, tens.n_legs()):
  #     print(f"Leg {leg}: dim {tens.dim_leg(leg)}")
  #   print(32*'-')
  # print(32*'-')

  return lhs, rhs