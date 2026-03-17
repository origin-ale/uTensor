import numpy as np
import tens_net as tn
import mps
from split_heisenberg import build_mpos
from tqdm import trange

N = 20
delta = 1e-3
n_steps = int(1/delta)
bond_dim = 2

state = mps.Mps(N=N)
mpo_even, mpo_odd = build_mpos(N, delta)

for i in trange(0,n_steps):
  state = mps.apply_mponn(mpo_even, state, bonddim_= bond_dim)
  state = mps.apply_mponn(mpo_odd, state, bonddim_= bond_dim)

print(state)