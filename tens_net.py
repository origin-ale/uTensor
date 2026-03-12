import numpy as np
import itertools as it

class LinkingError: Exception

class Tensor:
  def __init__(self, ndarr: np.ndarray | list):
    self.elements = np.array(ndarr)
    self.links = {}

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
         
  def bundle_legs(self, *legs):
    """Bundle a tuple of legs. The legs are bundled together at the lowest-numbered leg."""
    el = self.elements
    bundled_leg = min(legs)
    elm = np.moveaxis(el, legs, (0,1))
    elmp = np.concat(np.unstack(elm, axis=0), axis=0)
    self.elements = np.moveaxis(elmp, 0, bundled_leg)

  def unbundle_leg(self, leg, leg_dims):
    pass