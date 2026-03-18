import tens_net as tn
import mps
from split_heisenberg import build_site_heisenH
import tqdm
from copy import copy
import numpy as np

from scipy.linalg import expm

N = 4
np.set_printoptions(precision = 3, suppress=True)

print(16*'=', "Running exact evolution", 16*'=')
factors = [mps.MpsElement(1/np.sqrt(2) * (np.array([[[1],[0]]]) + (-1)**i * np.array([[[0],[1]]]))) for i in range(0,N)]
state = mps.Mps(init_factors=factors)
initial_mtx = state[0]
contract_steps = range(1, N)
for i in contract_steps:
  initial_mtx = tn.contract(copy(initial_mtx), state[i], initial_mtx.n_legs()-1, 0)
initial_mtx, partition_dims = tn.matrixize(copy(initial_mtx), (0,1,2,3,4))
initial_mtx.bundle_legs(0,1)
print("Initial state:", initial_mtx.elements)

H = np.zeros((2**N, 2**N))
for i in range(0,N):
  H += build_site_heisenH(N,i)

U = expm(- 1j * H)
final_mtx = U @ initial_mtx.elements

print("Final state:", final_mtx)