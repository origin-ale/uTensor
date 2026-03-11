import numpy as np

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
         
  def bundle_legs(self, *legs):
    """Bundle a tuple of legs. The legs are bundled together at the lowest-numbered leg."""
    el = self.elements
    old_legs = list(range(self.n_legs()))
    old_dims = [self.dim_leg(l) for l in old_legs]
    conserved_legs = [l for l in old_legs if l not in legs]
    if conserved_legs == []:
      self.elements = np.ravel(self.elements)
      return
    conserved_dims = [np.size(el, l) for l in conserved_legs]

    bundled = [
      [np.take(el, i, cl).flatten() for i in range(cd)]
      for (cd,cl) in zip(conserved_dims, conserved_legs)
    ]
    if max(legs) > conserved_legs[0]:
      stack_leg = conserved_legs[0]
    else:
      stack_leg = conserved_legs[0] - (len(legs)-1)

    self.elements = np.stack(bundled[0], stack_leg)