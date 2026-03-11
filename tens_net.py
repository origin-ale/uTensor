import numpy as np

class Tensor:
  def __init__(self, ndarr: np.ndarray):
    self.elements = ndarr

  def n_legs(self):
    return self.elements.ndim
  
  def dim_leg(self, leg: int):
      return self.elements.shape[leg]
         
    