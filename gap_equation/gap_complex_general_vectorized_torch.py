#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 14:42:23 2023

@author: user
"""
import numpy as np
from scipy.special import p_roots, cg_roots
import matplotlib.pyplot as plt
import gc
import time
from tqdm import tqdm
import os
import contextlib
import torch


##############################################################################
######################         Parameters         ############################
##############################################################################

# interaction parameters
model_name = "AWW" # MT or AWW or whatever, this is used only for filename, switch interaction manually !!!
omega      = 0.4   # MT: 0.3   0.4   0.5,   AWW: 0.35  0.4   0.45  0.5  0.55
dee        = 0.1     # MT: 1.24  0.93  0.744, AWW: 1.314 1.152 1.025 1.   0.84
N_f        = 4
gamma_m    = 12./(33-2*N_f)
tau        = np.exp(2.)-1.
Lambda_QCD = 0.234
m_t        = 0.5

# Set vertex; currently supports 'BC' and 'Dirac'
vertex='BC'

# set current-quark mass
m_q = 0.12  #  MT: 0.00374  0.125 , AWW: 0.005, 0.120, 1.125
# renormalization point
mu  = 19.**2

# Regularization
pvcut = 2.02e3

# numerical integration parameters
largest_q2 = 1.e3   # MT: 1.e6   AWW: 1.e3
n_q2 = 128
n_z  = 512  
n_y  = 1   
n_x  = 1  

# max number of iterations in iteration loop
MAX_ITERATIONS = 140
# relative accuracy needed for convergence
EPSILON = 1e-5

# Must split up grids due to memory constraints, i.e. can't do calc. fully vectorized.
# (e.g.: n_q2=32 (->n_p2=33), n_z=n_y=16 and n_x=8 needs almost 20GB of RAM if fully vectorized)
# Number of lists, that the q2,etc grids shall be split up into
# Set to 1 for full vectorization (no splitting) and n_q2,etc for fully sequential evaluation
num_p2_parts = 1
num_q2_parts = 1
num_z_parts  = 8
num_y_parts  = 1
num_x_parts  = 1
# Note also: almost all variables will still be vectorized with the indices having the following meaning:
#   [p2, q2,z,y,x, mu,nu, row,column]

# Under-Relaxation (set to 1 for no under-relaxation)
alpha = 1

# Plot setting
linewidth = 1

# Precision to be used (float should be sufficient if epsilon<=1e-5)
dtype  = torch.complex64#128

# Device to be used (cuda means GPU)
device = 'cuda'

##############################################################################
######################         Functions          ############################
##############################################################################

def tensor_expand_dims(tensor, dim):
    for ax in dim:
        tensor = torch.unsqueeze(tensor, ax)
    return tensor

def tensor_trace(tensor, dim1, dim2):
    return tensor.diagonal(offset=0, dim1=dim1, dim2=dim2).sum(dim=-1)

def create_index_lists(total_length, wanted_parts):
    """
    Create list containing 'wanted_parts' lists with numbers running from 0 to total_length-1 accross the lists
     Useful for partial vectorization, see AB_p function below and the for-loops inside it
     create_index_lists(9,1) -> [[0, 1, 2, 3, 4, 5, 6, 7, 8]]
     create_index_lists(9,3) -> [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
     create_index_lists(9,4) -> [[0, 1], [2, 3], [4, 5], [6, 7, 8]]
     create_index_lists(9,9) -> [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
     Motivated by: https://stackoverflow.com/questions/752308/split-list-into-smaller-lists-split-in-half#
    """
    alist = list(range(total_length))
    return [ alist[i*total_length // wanted_parts: (i+1)*total_length // wanted_parts] 
             for i in range(wanted_parts) ]

def pv_cut(q2):
    """
    Compute Pauli-Villars regularization cutoff
     scalar->scalar or torch.tensor->torch.tensor of same shape
    """
    return pvcut/(pvcut + q2)

# pre-compute potential factors
pot_fact_1 = 4 * torch.pi**2 * dee / omega**6
pot_fact_2 = 4 * torch.pi**2 * gamma_m
def interaction_MT(Q2):
    """
    preset MT interaction
     alpha
     scalar->scalar or torch.tensor->torch.tensor of same shape
    """
    return (pot_fact_1 * Q2**2 * torch.exp(-Q2 / omega**2)
          + pot_fact_2 * (1. - torch.exp(-Q2/(4.*m_t**2)))
          / (.5*torch.log(tau+(1.+Q2/Lambda_QCD**2)**2)))

def interaction_MTJ(Q2):
    """
    preset MTJ interaction
     scalar->scalar or torch.tensor->torch.tensor of same shape
    """
    return (pot_fact_1 * Q2**2 * torch.exp(-Q2 / omega**2)
          + pot_fact_2 * (1. - torch.exp(-Q2/(4.*m_t**2)))**2 
          / (.5*torch.log(tau+(1.+Q2/Lambda_QCD**2)**2)))

def interaction_AWW(Q2):
    """
    preset AWW interaction
     scalar->scalar or torch.tensor->torch.tensor of same shape
    """
    return pot_fact_1 * Q2**2 * torch.exp(-Q2 / omega**2)

def D_free(Q):
    """
    Q: torch.tensor of shape (*, *,*,*,*, 1,1, 4,1)
     Output: torch.tensor of shape (*, *,*,*,*, 4,4, 1,1)
    """
    Q2 = torch.sum(Q**2, dim=7)
    Q2 = tensor_expand_dims(Q2, dim=[7])

    Q_mu = torch.swapaxes(Q, 5,7)
    Q_nu = torch.swapaxes(Q, 6,7)
    
    identity = tensor_expand_dims(torch.eye(4, dtype=dtype, device=device), dim=(0, 1,2,3,4, 7,8))

    return (identity - Q_mu*Q_nu/Q2)/Q2

# Define Euclidean Dirac matrices
# np tensor of gammas 1-4
dirac_matrices = torch.tensor([
    # gamma1
    [[0, 0, 0, -1j],
    [0, 0, -1j, 0],
    [0, 1j, 0, 0],
    [1j, 0, 0, 0]], 
    # gamma2
    [[0, 0, 0, -1],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [-1, 0, 0, 0]], 
    # gamma3
    [[0, 0, -1j, 0],
    [0, 0, 0, 1j],
    [1j, 0, 0, 0],
    [0, -1j, 0, 0]], 
    # gamma4
    [[1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, -1]]
], device=device, dtype=dtype)
dirac_matrices_mu = tensor_expand_dims(dirac_matrices, dim=(0, 1,2,3,4, 6))
dirac_matrices_nu = tensor_expand_dims(dirac_matrices, dim=(0, 1,2,3,4, 5))

def slash(q):
    """
    q: torch.tensor of shape (*,4,1)
     Output: torch.tensor of shape (*,4,4)
    """
    return tensor_expand_dims(q[...,0,:],dim=[-2])* dirac_matrices[0] +\
           tensor_expand_dims(q[...,1,:],dim=[-2])* dirac_matrices[1] +\
           tensor_expand_dims(q[...,2,:],dim=[-2])* dirac_matrices[2] +\
           tensor_expand_dims(q[...,3,:],dim=[-2])* dirac_matrices[3] 
           
def S(q, A_q, B_q):
    """
    q: torch.tensor of shape (*, *,*,*,*, *,*, 4,1)
     A/B_q: torch.tensor of shape (*, *,*,*,*, *,*, 1,1)
     Output: torch.tensor of shape (*, *,*,*,*, *,*, 4,4)
    """
    q2 = torch.sum(q**2, dim=7)
    q2 = tensor_expand_dims(q2, dim=[7])
    mat = ((-1j)*A_q*slash(q) + B_q) / (q2*A_q**2 + B_q**2)
    return mat

# stole mapping implementation from Thomas' repository
def mom_map(n,c=.1,a=1,sup=False,mapping='graz'):
    """
    mom_map(x, NumPar)
    
    Maps the Gauss integration points (-1,1) to (0,infinity).
    Returns mapped tensor of mapping and derivative w.r.t. x as tuple
    ( p(x), dp(x)/dx ).
    
    - 'a' is modified according to 'SupMom' for 'graz'-mapping
    - if 'sup_mom'=NumPar['SupMom'] is given, 'a' is changed such that maximal momentum is met
    """
    x, w = p_roots(n)#Gauss-Legendre roots+weights
    
    if mapping in ('dubna','d'):
        if sup: a = np.log((sup-c)/(sup+c)) / np.log(x.max())
        mapping     = c*(1+x**a)/(1-x**a)
        derivative  = 2*c * (a*x**(a-1)) / (1-x**a)**2
    
    elif mapping in ('graz','g'):
        x_bar = (1.+x)/2
        if sup: a = np.log(1. + sup/c * (np.exp(1)-np.exp(x_bar.max()))) / x_bar.max()
        mapping     = c * (np.exp(a*x_bar)-1) / (np.exp(1)-np.exp(x_bar))
        derivative  = 0.5*( c*a*np.exp(a*x_bar) + mapping*np.exp(x_bar) ) / (np.exp(1)-np.exp(x_bar))
    
    return mapping, derivative, x, w

def q_fun(q2, z, y, x):
    """
    build 4-vector (q or p) from hyperbolic coordinates
     Shapes: q2: (1, n_q2,1,1,1, 1,1, 1,1) or (n_p2, 1,1,1,1, 1,1, 1,1)
             z:  (1, 1,n_z,1,1, 1,1, 1,1) or (1, 1,1,1,1, 1,1, 1,1)
             y:  (1, 1,1,n_y,1, 1,1, 1,1) or (1, 1,1,1,1, 1,1, 1,1)
             x:  (1, 1,1,1,n_x, 1,1, 1,1) or (1, 1,1,1,1, 1,1, 1,1)
     Output: (1, n_q2,n_z,n_y,n_x, 1,1, 4,1) or (n_p2, 1,1,1,1, 1,1, 4,1)
    """
    phi = torch.pi*(1+x)

    normalization = torch.sqrt(q2)

    q = normalization * torch.cat([
        torch.sqrt(1-z**2)*torch.sqrt(1-y**2)*torch.sin(phi),
        torch.sqrt(1-z**2)*torch.sqrt(1-y**2)*torch.cos(phi),
        torch.sqrt(1-z**2)*y+0*x, #the "+0*x is only there to fix the dimensions
        z +0*x+0*y], dim=7)
    return q


##############################################################################
######################         Execution          ############################
##############################################################################
if __name__ == '__main__':

    # create output directory and filenames
    directory_name = model_name+"_"+str(omega)+"_"+str(dee)+"_"+str(m_q)+"_"+vertex+"_"+str(n_q2)+"_"+str(n_z)+"_"+str(n_y)+"_"+str(n_x)
    real_filename = "_"
    # create a dir per run
    if not os.path.exists(directory_name):
        os.mkdir(directory_name, mode=0o755)
    
    
    ##############################################################################
    
    # Prepare integration grids
    # get radial/q2 integration grid, Gauss-Legendre+Graz Mapping
    q2_grid, dq2_ds_grid, s_grid, q2_weights = [torch.from_numpy(item).to(dtype).to(device=device) for item in mom_map(n_q2, sup=largest_q2)]
    # get z integration grid, Gauss-Chebyshev
    z_grid, z_weights = [torch.from_numpy(item).to(dtype).to(device=device) for item in cg_roots(n_z, alpha=1)]
    # get y integration grid, Gauss-Legendre
    y_grid, y_weights = [torch.from_numpy(item).to(dtype).to(device=device) for item in p_roots(n_y)]
    # get x integration grid, Gauss-Legendre
    x_grid, x_weights = [torch.from_numpy(item).to(dtype).to(device=device) for item in p_roots(n_x)]  # will be split up and used piece by piece due to memory constraints
    #Add mu to front of p2 grid
    p2_grid = torch.hstack([torch.tensor([mu], device=device, dtype=dtype), q2_grid]).reshape(-1)
    n_p2    = len(p2_grid)     
    
    # Vectorize
    q2_grid_vec     = tensor_expand_dims(q2_grid,     dim=(0, 2,3,4, 5,6, 7,8))
    dq2_ds_grid_vec = tensor_expand_dims(dq2_ds_grid, dim=(0, 2,3,4, 5,6, 7,8))
    s_grid_vec      = tensor_expand_dims(s_grid,      dim=(0, 2,3,4, 5,6, 7,8))
    q2_weights_vec  = tensor_expand_dims(q2_weights,  dim=(0, 2,3,4, 5,6, 7,8))
    z_grid_vec      = tensor_expand_dims(z_grid,      dim=(0, 1,3,4, 5,6, 7,8))
    z_weights_vec   = tensor_expand_dims(z_weights,   dim=(0, 1,3,4, 5,6, 7,8))
    y_grid_vec      = tensor_expand_dims(y_grid,      dim=(0, 1,2,4, 5,6, 7,8))
    y_weights_vec   = tensor_expand_dims(y_weights,   dim=(0, 1,2,4, 5,6, 7,8))
    x_grid_vec      = tensor_expand_dims(x_grid,      dim=(0, 1,2,3, 5,6, 7,8))
    x_weights_vec   = tensor_expand_dims(x_weights,   dim=(0, 1,2,3, 5,6, 7,8))
    p2_grid_vec     = tensor_expand_dims(p2_grid,     dim=( 1,2,3,4, 5,6, 7,8))
    
    # Create index lists for the (partial) vectorization
    p2_index_lists = create_index_lists(n_p2, num_p2_parts)
    q2_index_lists = create_index_lists(n_q2, num_q2_parts)
    z_index_lists  = create_index_lists(n_z,  num_z_parts)
    y_index_lists  = create_index_lists(n_y,  num_y_parts)
    x_index_lists  = create_index_lists(n_x,  num_x_parts)
    
    
    def AB_p(A_sol_prev, B_sol_prev):
        """
        Calculate A and B without renormalization (only prefactor*integral part of rhs)
         A_sol_prev, B_sol_prev: 1D tensors of shape (n_p,)
         returns two tensors of shape (n_p,)
        """
        A_sol_prev = A_sol_prev.reshape(-1)
        A_p_prev   = tensor_expand_dims(A_sol_prev, dim=(1,2,3,4, 5,6, 7,8))
        A_q_prev   = tensor_expand_dims(A_sol_prev[1:], dim=(0, 2,3,4, 5,6, 7,8)) #mu is not part of the q2 grid (but zeroth element of p2 grid)
        B_sol_prev = B_sol_prev.reshape(-1)
        B_p_prev   = tensor_expand_dims(B_sol_prev, dim=(1,2,3,4, 5,6, 7,8))
        B_q_prev   = tensor_expand_dims(B_sol_prev[1:], dim=(0, 2,3,4, 5,6, 7,8)) 
        
        # Calculate Gradients of A and B on q2_grid (needed for BC vertex)
        A_sol_prev_gradient = torch.gradient(A_q_prev.reshape(-1), spacing=[q2_grid])[0]
        B_sol_prev_gradient = torch.gradient(B_q_prev.reshape(-1), spacing=[q2_grid])[0]
        A_sol_prev_gradient = torch.cat([torch.tensor([torch.nan + 1j*torch.nan], dtype=dtype, device=device), A_sol_prev_gradient]) #Gradient at mu is not needed, since mu is not part of the q2-grid
        B_sol_prev_gradient = torch.cat([torch.tensor([torch.nan + 1j*torch.nan], dtype=dtype, device=device), B_sol_prev_gradient])
        A_p_prev_gradient   = tensor_expand_dims(A_sol_prev_gradient, dim=(1,2,3,4, 5,6, 7,8))
        B_p_prev_gradient   = tensor_expand_dims(B_sol_prev_gradient, dim=(1,2,3,4, 5,6, 7,8))
        A_p_prev_gradient   = torch.tile(A_p_prev_gradient, (1, n_q2,1,1,1, 1,1, 1,1))  # needed for vectorized version of BC
        B_p_prev_gradient   = torch.tile(B_p_prev_gradient, (1, n_q2,1,1,1, 1,1, 1,1))
        
        A_p_list   = []
        B_p_list   = []
        for p2_index in p2_index_lists:
            p2         = p2_grid_vec[p2_index,...]
            p          = q_fun(p2, torch.ones(p2.shape, dtype=dtype, device=device), torch.zeros(p2.shape, dtype=dtype, device=device), torch.zeros(p2.shape, dtype=dtype, device=device))  # simple choice of p that satisfies p*p=p2 
            p_slash    = slash(p)
            
            A_p        = A_p_prev[p2_index,...]
            B_p        = B_p_prev[p2_index,...]
            
            integral_A = torch.zeros(len(p2_index), dtype=dtype, device=device)
            integral_A = tensor_expand_dims(integral_A, dim=(1,2,3,4, 5,6, 7,8))
            integral_B = torch.zeros(len(p2_index), dtype=dtype, device=device)
            integral_B = tensor_expand_dims(integral_B, dim=(1,2,3,4, 5,6, 7,8))
            for q2_index in q2_index_lists:
                q2           = q2_grid_vec[:,q2_index,...]
                dq2_ds       = dq2_ds_grid_vec[:,q2_index,...]
                q2_weight    = q2_weights_vec[:,q2_index,...]
                
                A_q          = A_q_prev[:,q2_index,...]
                B_q          = B_q_prev[:,q2_index,...]
                A_p_gradient = A_p_prev_gradient[p2_index,...][:,q2_index,...]
                B_p_gradient = B_p_prev_gradient[p2_index,...][:,q2_index,...]
                for z_index in z_index_lists:
                    z        = z_grid_vec[:, :,z_index,...]
                    z_weight = z_weights_vec[:, :,z_index,...]
                    for y_index in y_index_lists:
                        y        = y_grid_vec[:, :,:,y_index,...]
                        y_weight = y_weights_vec[:, :,:,y_index,...]
                        for x_index in x_index_lists:   
                            x        = x_grid_vec[:, :,:,:,x_index,...]
                            x_weight = x_weights_vec[:, :,:,:,x_index,...]
         
                            q        = q_fun(q2,z,y,x)
                            Q        = p-q
                            Q2       = torch.sum(Q**2, dim=7)
                            Q2       = tensor_expand_dims(Q2, dim=[7])
                            
                            D_matrix = D_free(Q)
                            del Q
                            gc.collect()
                            S_matrix = S(q, A_q, B_q) 
                            S_matrix = S_matrix
                            
                            if vertex=='BC':
                                # Ball-Chiu Vertex
                                Sigma_A  = (A_q + A_p)/2
                                with contextlib.redirect_stderr(None): # suppress "true divide" errors
                                    Delta_A  = (A_q - A_p)/(q2-p2)
                                    Delta_B  = (B_q - B_p)/(q2-p2)
                                Delta_A[torch.isnan(Delta_A)] = A_p_gradient[torch.isnan(Delta_A)] #in case q2==p2, Delta is the differential quotient
                                Delta_B[torch.isnan(Delta_B)] = B_p_gradient[torch.isnan(Delta_B)]
                                l        = (q+p)/2
                                l_slash  = slash(l)
                                l_nu     = torch.swapaxes(l, 6,7)
                                Gamma_nu = Sigma_A*dirac_matrices_nu + 2*l_nu*(l_slash*Delta_A-1j*Delta_B)
                                Gamma_nu = Gamma_nu
                            elif vertex=='Dirac':
                                # Old code used this vertex instead of BC
                                Gamma_nu = dirac_matrices_nu 
                            else:
                                raise ValueError("Vertex not supported!")
                            del q
                            gc.collect()
    
                            matrix_A    = torch.sum(D_matrix * p_slash @ dirac_matrices_mu @ S_matrix @ Gamma_nu, dim=(5,6)) #sum over mu and nu
                            matrix_A    = tensor_expand_dims(matrix_A, dim=(5,6))
                            tr_matrix_A = tensor_trace(matrix_A, dim1=7,dim2=8) # trace over matrix 
                            del matrix_A
                            gc.collect()
                            tr_matrix_A = tensor_expand_dims(tr_matrix_A, dim=(7,8)) # last 4 indices are placeholders to take care of tensor's shape
                            integral_A_contribution = torch.sum(q2_weight*dq2_ds*q2 * z_weight*y_weight*x_weight \
                                *interaction_AWW(Q2) * (4/3) * tr_matrix_A *pv_cut(Q2), dim=(1,2,3,4, 5,6, 7,8)) #sum over dim1-4 is integration (sums over 5-8 are due to vectorization, there is actually only one element, i.e no sum)
                            del tr_matrix_A
                            gc.collect()
                            integral_A_contribution = tensor_expand_dims(integral_A_contribution, dim=(1,2,3,4, 5,6, 7,8))
                            integral_A             += integral_A_contribution
                            
                            matrix_B    = torch.sum(D_matrix * dirac_matrices_mu @ S_matrix @ Gamma_nu, dim=(5,6))
                            del D_matrix
                            del S_matrix
                            del Gamma_nu
                            gc.collect()
                            matrix_B    = tensor_expand_dims(matrix_B, dim=(5,6))
                            tr_matrix_B = tensor_trace(matrix_B, dim1=-2,dim2=-1) 
                            del matrix_B
                            gc.collect()
                            tr_matrix_B = tensor_expand_dims(tr_matrix_B, dim=(7,8)) 
                            integral_B_contribution = torch.sum(q2_weight*dq2_ds*q2 * z_weight*y_weight*x_weight \
                                *interaction_AWW(Q2) * (4/3) * tr_matrix_B *pv_cut(Q2), dim=(1,2,3,4, 5,6, 7,8))
                            del Q2
                            del tr_matrix_B
                            gc.collect()
                            integral_B_contribution = tensor_expand_dims(integral_B_contribution, dim=(1,2,3,4, 5,6, 7,8))
                            integral_B             += integral_B_contribution  
                            
            integral_A *= (1/(2*torch.pi)**4) * (1/2) *torch.pi   # integral prefactors from coordinate changes
            integral_B *= (1/(2*torch.pi)**4) * (1/2) *torch.pi       
            A_p_part    = 1 - 1j/(4*p2)*integral_A  
            B_p_part    = (1/4)*integral_B 
            A_p_list.append(A_p_part)
            B_p_list.append(B_p_part)

        # Transform to one 1D arrays and return them
        A_p_tensor  = torch.cat(A_p_list).reshape(-1)
        B_p_tensor  = torch.cat(B_p_list).reshape(-1)

        return A_p_tensor, B_p_tensor
    
    # set initial guess
    A_sol_prev = 1.5 * torch.ones(n_p2, dtype=dtype, device=device)
    B_sol_prev = max(m_q, 0.00001) * torch.ones(n_p2, dtype=dtype, device=device)
    
    for iteration_index in tqdm(range(MAX_ITERATIONS)):
        A_sol, B_sol = AB_p(A_sol_prev, B_sol_prev)    
        A_mu = A_sol[0]  
        B_mu = B_sol[0]
    
        #Renormalization (except for zeroth position , which is mu itself)
        A_sol[1:] = 1-A_mu + A_sol[1:]    
        B_sol[1:] = m_q-B_mu + B_sol[1:]   
        
        # Save each step, incase PC crashes
        np.savetxt(os.path.join(directory_name, real_filename+"prop_tmp.txt"), np.column_stack((p2_grid.cpu().numpy(), B_sol.cpu().numpy()/A_sol.cpu().numpy(), A_sol.cpu().numpy(), B_sol.cpu().numpy())))

        # check for convergence
        with contextlib.redirect_stderr(None): # suppress "true divide" errors
            re_diff_A = torch.abs((A_sol - A_sol_prev)/A_sol_prev)
            re_diff_B = torch.abs((B_sol - B_sol_prev)/B_sol_prev)
        re_diff_A[torch.isnan(re_diff_A)] = 0
        re_diff_B[torch.isnan(re_diff_B)] = 0
        re_diff_A                         = torch.max(re_diff_A).item()
        re_diff_B                         = torch.max(re_diff_B).item()
        # print diagnostic output
        print("Iteration: ", iteration_index, A_mu, re_diff_A, B_mu, re_diff_B)
    
        if (re_diff_A < EPSILON) and (re_diff_B < EPSILON):
            print("Convergence reached after", iteration_index, "iterations.")
            break
    
        # hand (under-relaxed) solutions over for next interation step's convergence check
        A_sol_prev = alpha*A_sol + (1-alpha)*A_sol_prev
        B_sol_prev = alpha*B_sol + (1-alpha)*B_sol_prev
    
    A_sol   = A_sol.cpu().numpy()
    B_sol   = B_sol.cpu().numpy()
    q2_grid = q2_grid.cpu().numpy()
    A_sol   = A_sol[1:] #throw away mu
    B_sol   = B_sol[1:]
    A_sol   = np.real(A_sol) # imaginary parts are just numeric noise
    B_sol   = np.real(B_sol)
    q2_grid = np.real(q2_grid)
    
    fig = plt.figure()
    plt.plot(q2_grid, A_sol, "-", label=r"$A(p^2)$",linewidth=linewidth)
    plt.plot(q2_grid, B_sol, "-", label=r"$B(p^2)$",linewidth=linewidth)
    plt.plot(q2_grid, B_sol/A_sol, "-", label=r"$M(p^2)$",linewidth=linewidth)
    
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$p^2\,[GeV^2]$")
    plt.legend(loc=3)
    
    np.savetxt(os.path.join(directory_name, real_filename+"prop.txt"), np.column_stack((q2_grid, B_sol/A_sol, A_sol, B_sol)))
    plt.savefig(os.path.join(directory_name, real_filename+"realaxis.pdf"))
    plt.close()















