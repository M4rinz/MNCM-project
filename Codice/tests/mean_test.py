import os
import sys

# Launch this script from within the test/ folder!
sys.path.append(os.path.abspath(os.path.join('..')))

from utilities.data import generate_random_P, create_observations
from utilities.num_methods import compute_stationary_LU_GTH
from utilities.estimators import P_mom_stationary

import numpy as np

S, D = 10, 0.5
SEED = 42

P = generate_random_P(S, 'dirichlet', precision=D, seed=SEED)
pi = compute_stationary_LU_GTH(P=P)
print("pi =")
print(pi)
print()

# Let's reproduce an "unfavourable setting"
T = 10	# n° of total timesteps
K = 1	# n° of total repetitions
N = 100

n_t_array, y_t_array, A = create_observations(T=T, K=K, N=N, 
											  pi_0=pi, stationary=True,
											  noise_type='gaussian', stdev=1, seed=SEED)
print(f"n_t_array.shape = {n_t_array.shape}.\t (T,K,S) = ({T}, {K}, {S})")
print()

mean_original = n_t_array.mean(axis=(0,1))
print("Mean of the original data over the temporal (T) and repetitions (K) dimension:")
print(mean_original)
print(f"Scaling by population size N = {N} yields:")
print(mean_original/N)
#print(f"pi = {pi}")
print()

distance = np.linalg.norm(pi - mean_original/N, 2)
print(f"norm(pi - mean/N)_2 = {distance}")

