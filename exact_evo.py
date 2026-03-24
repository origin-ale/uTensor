import tens_net as tn
import mps
from split_heisenberg import build_site_heisenH
import tqdm
from copy import copy
import numpy as np

from scipy.linalg import expm

N = 5
np.set_printoptions(precision = 3, suppress=True)

def mps_to_vector(state: mps.Mps):
  """Convert MPS to state vector, permuting physical indices so that meaning is the same as tebd.py"""
  n_sites = len(state)

  out = state[0]
  for i in range(1, n_sites):
    out = tn.contract(copy(out), state[i], out.n_legs()-1, 0)
  
  elements = np.asarray(out.elements)
  
  if n_sites % 2 == 1:
    axperm = [(i+(n_sites-1))%n_sites + 1 for i in range(n_sites)]
    axperm.append(n_sites+1)
    axperm.insert(0, 0)
    elements = np.transpose(elements, axperm)

  return elements.flatten()

def build_basis_permutation(n_sites: int):
  """Build permutation matrix mapping computational basis to MPS contraction basis."""
  dim = 2**n_sites
  perm = np.zeros((dim, dim))
  for k in range(dim):
    bits = [(k >> (n_sites-1-j)) & 1 for j in range(n_sites)]
    factors = [mps.MpsElement(np.array([[[1],[0]]])) if b == 0 else mps.MpsElement(np.array([[[0],[1]]])) for b in bits]
    basis_state = mps.Mps(init_factors=factors)
    vec = mps_to_vector(basis_state)
    idx = int(np.argmax(np.abs(vec)))
    perm[idx, k] = 1
  return perm

print(16*'=', "Running exact evolution", 16*'=')
factors = [mps.MpsElement(1/np.sqrt(5) * (np.array([[[2],[0]]]) + (-1)**i * np.array([[[0],[1]]]))) for i in range(0,N)]
# factors = [mps.MpsElement(1/np.sqrt(2) * (np.array([[[1],[0]]]) + (-1)**i * np.array([[[0],[1]]]))) for i in range(0,N)]
# factors = [mps.MpsElement(np.array([[[1],[0]]])) for i in range(0,N)]
state = mps.Mps(init_factors=factors)
initial_mtx = tn.Tensor(mps_to_vector(state))
print("Initial state:", initial_mtx.elements)
print("with norm", np.linalg.norm(initial_mtx.elements))

H_comp = np.zeros((2**N, 2**N))
for i in range(0,N):
  H_comp += build_site_heisenH(N,i)

perm = build_basis_permutation(N)
H = perm @ H_comp @ perm.T

U = expm(- 1j * H)
final_mtx = U @ initial_mtx.elements

print("\nFinal state:", final_mtx)
print("with norm", np.linalg.norm(final_mtx),'\n')