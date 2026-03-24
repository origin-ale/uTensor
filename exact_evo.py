import tens_net as tn
import mps
from split_heisenberg import build_site_heisenH
import tqdm
from copy import copy
import numpy as np

from scipy.linalg import expm

N = 6
np.set_printoptions(precision = 3, suppress=True)

def mps_to_vector(state: mps.Mps):
  """Convert MPS to state vector"""
  n_sites = len(state)

  out = state[0]
  for i in range(1, n_sites):
    out = tn.contract(copy(out), state[i], out.n_legs()-1, 0)
  
  elements = np.asarray(out.elements)

  return elements.flatten()

print(16*'=', "Running exact evolution", 16*'=')
factors = [mps.MpsElement(1/np.sqrt(5) * (np.array([[[2],[0]]]) + (-1)**i * np.array([[[0],[1]]]))) for i in range(0,N)]
# factors = [mps.MpsElement(1/np.sqrt(2) * (np.array([[[1],[0]]]) + (-1)**i * np.array([[[0],[1]]]))) for i in range(0,N)]
# factors = [mps.MpsElement(np.array([[[1],[0]]])) for i in range(0,N)]
state = mps.Mps(init_factors=factors)
initial_mtx = tn.Tensor(mps_to_vector(state))
print("Initial state:", initial_mtx.elements)
print("with norm", np.linalg.norm(initial_mtx.elements))

H = np.zeros((2**N, 2**N))
for i in range(0,N):
  H += build_site_heisenH(N,i)

U = expm(- 1j * H)
final_mtx = U @ initial_mtx.elements

print("\nFinal state:", final_mtx)
print("with norm", np.linalg.norm(final_mtx),'\n')