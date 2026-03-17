import tens_net as tn
import mps
from split_heisenberg import build_mpos
import tqdm
from copy import copy
import numpy as np

N = 4
delta = 1e-3
n_steps = int(1/delta)
bond_dim = 10

state = mps.Mps(N=N)
mpo_even, mpo_odd = build_mpos(N, delta)

evo_steps = tqdm.trange(0,n_steps)
for i in evo_steps:
  print_process = False
  evo_steps.set_description("Evolving system")
  if i in (1,5,10, n_steps-1): print_process = True
  state = mps.apply_mponn(mpo_even, state, bonddim_= bond_dim)
  state = mps.apply_mponn(mpo_odd, state, bonddim_= bond_dim)

final_mtx = state[0]
contract_steps = range(1, N-1)
for i in contract_steps:
  final_mtx = tn.contract(copy(final_mtx), state[i], final_mtx.n_legs()-1, 0)

np.set_printoptions(precision = 3, suppress=True)
print(tn.matrixize(final_mtx, (2,3))[0].elements)