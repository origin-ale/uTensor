import numpy as np
import tens_net as tn
import mps
from scipy.linalg import expm

def get_bit(n: int, i: int):
  """
  Get the ith bit of an int n
  """
  return n >> i & 1

def flip_bits(n: int, i: int, j: int):
  """
  Flip (not switch) the ith and jth bits of an int n
  """
  return n ^ (2**i + 2**j)

def build_site_heisenH(N: int, i: int):
  """
  Build the Hamiltonian matrix for NN interaction a Heisenberg spin chain

  Parameters
  ----------
  N : int
    The length of the spin chain (number of spins).
  i : int
    The site for which to build the hamiltonian
  """
  try: assert i < N
  except AssertionError: raise ValueError("j must be < N")
  state_n = 2**N
  hi = np.zeros((state_n, state_n))
  for s in range(0, state_n):
    j = (i+1)%N # rightward nearest neighbor of i, with periodicity enforced
    if get_bit(s,i) == get_bit(s,j): 
      hi[s,s] += .25
    else:
      hi[s,s] += -.25
      sp = flip_bits(s,i,j)
      hi[s,sp] = .5
  return hi

def build_mpos(N: int, delta: float):
  mpo_even = mps.MpoNN()
  mpo_odd = mps.MpoNN()
  state_n = 2**N
  for i in range(0, state_n, 2):
    mpo_even[len(mpo_even):len(mpo_even)] = expm(1j * delta * build_site_heisenH(N, i))
  for i in range(1, state_n, 2):
    mpo_odd[len(mpo_odd):len(mpo_odd)] = expm(1j * delta * build_site_heisenH(N, i))
  return mpo_even, mpo_odd