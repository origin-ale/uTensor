import numpy as np
from scipy.linalg import expm
from copy import copy
import tqdm
import sys

class MPS:
  def __init__(self, mpsTensors):
    self.gammas = mpsTensors # ax0 is the left bond, ax1 is the physical index, and ax2 is the right bond.
      
  def bond_contract_and_svd(self, bond, chi, move_right):
    """
    bond ... is 1-indexed, i.e. bond n connects tensor n-1 and n
    chi ... is the bond dimension to which we should truncate
    move_right ... if true the singular values are contracted with the right tensor, otherwise with the left tensor.
    """
    tens_l = copy(self.gammas[bond-1])     
    tens_r = copy(self.gammas[bond])
    init_shape_l = tens_l.shape
    init_shape_r = tens_r.shape
    assert init_shape_l[2] == init_shape_r[0]
    tens_l = tens_l.reshape(init_shape_l[0]*init_shape_l[1], init_shape_l[2])
    tens_r = tens_r.reshape(init_shape_r[0], init_shape_r[1]*init_shape_r[2])
    
    tens_lr = tens_l@tens_r
    U, S, Vh = np.linalg.svd(tens_lr, full_matrices=False)
    zero_mask = (S == 0)
    S = np.delete(S, zero_mask)
    U = np.delete(U, zero_mask, -1)
    Vh = np.delete(Vh, zero_mask, -2)
    chi_eff = min(len(S),chi)
    if move_right:
      tens_l = U[:,:chi_eff]
      tens_r = np.diag(S[:chi_eff]) @ Vh[:chi_eff,:]
    else:
      tens_l = U[:,:chi_eff] @ np.diag(S[:chi_eff])
      tens_r = Vh[:chi_eff,:]

    self.gammas[bond-1] = tens_l.reshape(init_shape_l[0], init_shape_l[1], chi_eff)
    self.gammas[bond]   = tens_r.reshape(chi_eff, init_shape_r[1], init_shape_r[2])
    return None

  def sweep(self, chi):
    N = len(self.gammas)
    for n in range(1,N):
      self.bond_contract_and_svd(n,chi,True)
    for n in range(N-1,0,-1):
      self.bond_contract_and_svd(n,chi,False)
    return None
  
  def gate_contract_and_svd(self, _gate, bond, chi, move_right):
    tens_l = copy(self.gammas[bond-1])        
    tens_r = copy(self.gammas[bond])
    gate = _gate
    gate_shape = gate.shape
    init_shape_l = tens_l.shape
    init_shape_r = tens_r.shape
    # gate indices are ordered as [s1_out, s2_out, s1_in, s2_in]
    assert init_shape_l[1] == gate_shape[2]
    assert init_shape_r[1] == gate_shape[3]
    # Build two-site tensor theta[l, s1, s2, r], apply gate G[s1', s2', s1, s2],
    # then matrixize as (l,s1') x (s2',r)
    theta = np.einsum('lsm,mtr->lstr', tens_l, tens_r)
    theta = np.einsum('abst,lstr->labr', gate, theta)
    c3 = theta.reshape(init_shape_l[0] * gate_shape[0], gate_shape[1] * init_shape_r[2])
    # print("Matrixized contraction 3 =")
    # print(c3)
    # print()

    U, S, Vh = np.linalg.svd(c3, full_matrices=False)
    zero_mask = (S == 0)
    S = np.delete(S, zero_mask)
    U = np.delete(U, zero_mask, -1)
    Vh = np.delete(Vh, zero_mask, -2)
    chi_eff = min(len(S),chi)
    if move_right:
      tens_l = U[:,:chi_eff]
      tens_r = np.diag(S[:chi_eff]) @ Vh[:chi_eff,:]
    else:
      tens_l = U[:,:chi_eff] @ np.diag(S[:chi_eff])
      tens_r = Vh[:chi_eff,:]

    self.gammas[bond-1] = copy(tens_l.reshape(init_shape_l[0], init_shape_l[1], chi_eff))
    self.gammas[bond]   = copy(tens_r.reshape(chi_eff, init_shape_r[1], init_shape_r[2]))

def build_site_heisenH_unitary(delta: float):
  mtx = np.array([
    [.25,  0,    0,    0 ],
    [0,   -.25,  .5,   0 ], 
    [0,   .5,   -.25,  0 ], 
    [0,   0,    0,    .25]
  ])
  mtx_exp = expm(- 1j * delta * mtx)
  mtx_exp = mtx_exp.reshape(2,2,2,2)
  return mtx_exp
  
N = 6
delta = 1e-3
n_steps = int(1/delta)
bond_dim = 10

np.set_printoptions(precision = 3, suppress=True)

print(16*'=', "Running TEBD with numpy", 16*'=')
factors = [1/np.sqrt(5) * (np.array([[[2],[0]]]) + (-1)**i * np.array([[[0],[1]]])) for i in range(0,N)]
# factors = [1/np.sqrt(2) * (np.array([[[1],[0]]]) + (-1)**i * np.array([[[0],[1]]])) for i in range(0,N)]
# factors = [np.array([[[1],[0]]]) for i in range(0,N)]
mps = MPS(factors)

vec = mps.gammas[0].copy()
for i in range(1, N):
  vec_dims = [vec.shape[l] for l in range(0, vec.ndim)]
  dim_prod = np.prod(vec_dims[:-1])
  vec = vec.reshape(dim_prod, vec_dims[-1])
  new = mps.gammas[i].copy()
  new = new.reshape(new.shape[0], new.shape[1]*new.shape[2])
  vec = vec @ new
  vec = vec.reshape(*vec_dims[:-1], mps.gammas[i].shape[1], mps.gammas[i].shape[2])
# if N%2 == 1:
#   axperm = [(i+(N-1))%N + 1 for i in range(0,N)]
#   axperm.append(N+1)
#   axperm.insert(0,0)
#   vec = np.transpose(vec, axperm)
print("Initial state:", vec.flatten())
print("with norm", np.linalg.norm(vec))

evo_steps = tqdm.trange(0,n_steps)
evo_steps.set_description(f"Evolving system")
for s in evo_steps:
  for i in range(1, N, 2):
    gate = build_site_heisenH_unitary(delta)
    mps.gate_contract_and_svd(gate, i, bond_dim, True)
  for i in range(2,N,2):
    gate = build_site_heisenH_unitary(delta)
    mps.gate_contract_and_svd(gate, i, bond_dim, True)
  mps.sweep(bond_dim)

print(*[t.shape for t in mps.gammas], sep = '--')

vec = mps.gammas[0].copy()
for i in range(1, N):
  vec_dims = [vec.shape[l] for l in range(0, vec.ndim)]
  dim_prod = np.prod(vec_dims[:-1])
  vec = vec.reshape(dim_prod, vec_dims[-1])
  new = mps.gammas[i].copy()
  new = new.reshape(new.shape[0], new.shape[1]*new.shape[2])
  vec = vec @ new
  vec = vec.reshape(*vec_dims[:-1], mps.gammas[i].shape[1], mps.gammas[i].shape[2])
# if N%2 == 1:
#   vec = np.transpose(vec, axperm)
print("\nFinal state:", vec.flatten())
print("with norm", np.linalg.norm(vec))