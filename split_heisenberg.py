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

def build_site_heisenH_unitary(delta: float):
  mtx = np.array([
    [.5,  0,    0,    0 ],
    [0,   -.5,  .5,   0 ],
    [0,   .5,   -.5,  0 ],
    [0,   0,    0,    .5]
  ])
  mtx_exp = expm(- 1j * delta * mtx)
  uni = tn.Tensor(mtx_exp)
  uni.unbundle_leg(1, (2,2))
  uni.unbundle_leg(0, (2,2))
  return mps.Unitary(uni.elements)

def build_tebd_mpos(N: int, delta: float):
  mpo_even = mps.Mpo()
  mpo_odd = mps.Mpo()
  for i in range(0, N, 2):
    uni = build_site_heisenH_unitary(delta)
    if i+1 < N:
      mpo_even.append(mps.AppliedUnitary(uni, (i,i+1)))
  for i in range(1, N, 2):
    uni = build_site_heisenH_unitary(delta)
    if i+1 < N:
      mpo_odd.append(mps.AppliedUnitary(uni, (i,i+1)))
  return mpo_even, mpo_odd