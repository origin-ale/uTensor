import tens_net as tn
import mps
from split_heisenberg import build_tebd_mpos
import tqdm
from copy import copy
import numpy as np

N = 4
delta = 1e-4
n_steps = int(1/delta)
bond_dim = 10

np.set_printoptions(precision = 3, suppress=True)

print(16*'=', "Running TEBD", 16*'=')
factors = [tn.Tensor(1/np.sqrt(2) * (np.array([[[1],[0]]]) + (-1)**i * np.array([[[0],[1]]]))) for i in range(0,N)]
state = mps.Mps(init_factors=factors)
initial_mtx = state[0]
contract_steps = range(1, N)
for i in contract_steps:
  initial_mtx = tn.contract(copy(initial_mtx), state[i], initial_mtx.n_legs()-1, 0)
initial_mtx, partition_dims = tn.matrixize(copy(initial_mtx), (0,1,2,3,4))
initial_mtx.bundle_legs(0,1)
print("Initial state:", initial_mtx.elements)

mpo_even, mpo_odd = build_tebd_mpos(N, delta)

evo_steps = tqdm.trange(0,n_steps)
evo_steps.set_description(f"Evolving system")

for i in evo_steps:
  print_process = False
  if i in (1,5,10, n_steps-1): print_process = True
  state = mps.apply_mpo(mpo_even, state, bonddim_= bond_dim)
  state = mps.apply_mpo(mpo_odd, state, bonddim_= bond_dim)

final_mtx = state[0]
contract_steps = range(1, N)
for i in contract_steps:
  final_mtx = tn.contract(copy(final_mtx), state[i], final_mtx.n_legs()-1, 0)

final_mtx, partition_dims = tn.matrixize(copy(final_mtx), (0,1,2,3,4))
final_mtx.bundle_legs(0,1)
print("Final state:", final_mtx.elements)