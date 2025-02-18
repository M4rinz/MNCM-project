import os
import sys

# Launch this script from within the test/ folder!
sys.path.append(os.path.abspath(os.path.join('..')))

from utilities.data import generate_random_P, create_observations
from utilities.num_methods import compute_stationary_LU_GTH
from utilities.estimators import P_mom_stationary

import numpy as np

def error_computation(M:np.ndarray) -> float:
	return np.linalg.norm(M - P, 'fro')/(S**2)

## Ok let's start

S, D = 10, 0.5
SEED = 42

P = generate_random_P(S, 'dirichlet', precision=D, seed=SEED)
pi = compute_stationary_LU_GTH(P=P)
print("pi =")
print(pi)
print()

# Let's reproduce an "unfavourable setting"
#T = 10	# n° of total timesteps
#K = 1	# n° of total repetitions
T = 10000
K = 20
N = 100

# Change here to change the variance
variance = 0

n_t_array, y_t_array, A = create_observations(T=T, K=K, N=N, 
											  pi_0=pi, stationary=True,
											  noise_type='gaussian', stdev=np.sqrt(variance), seed=SEED)
print(f"n_t_array.shape = {n_t_array.shape}.\t (T,K,S) = ({T}, {K}, {S})")
print()

P_mom, mu, Sigma = P_mom_stationary(y_array=y_t_array, A=A, N=N)

print("Mean as returned by the algorithm (avg across T (temporal) and K (repetitions) dimensions of the noisy data, normalized etc.):")
print(mu)
distance = np.linalg.norm(pi - mu, 2)
print(f"norm(pi - mean/N)_2 = {distance}")
print()
print("Mean as obtained from the original data (avg across T (temporal) and K (repetitions) dimensions of the original data):")
mean_original = n_t_array.mean(axis=(0,1))
print(mean_original)
distance = np.linalg.norm(pi - mean_original/N, 2)
print(f"norm(pi - mean_original/N)_2 = {distance}")
print()
print()

print(f"Distance between P and P_MoM (frobenius norm of the difference / {S}^2):")
print(f"{error_computation(P_mom)}")
print()
print()

# What if we used the invariant distribution in the algorithm?
m_t_hat = y_t_array.mean(axis=(0,1))
m_t_hat = np.linalg.solve(A,m_t_hat)

mu_hat = np.linalg.solve(A,pi)
mu_hat = mu_hat/np.linalg.norm(mu_hat,1)	# pleonastic

deviations = y_t_array - N*mu_hat #change to mu_hat?

Sigma_hat = np.matmul(deviations[:-1].transpose(0,2,1),deviations[1:]).mean(axis=0)/K
Sigma_hat = np.linalg.solve(A,Sigma_hat)
Sigma_hat = np.linalg.solve(A,Sigma_hat.T).T

P_mom_hat = P_mom = ((Sigma_hat/N+np.outer(mu_hat,mu_hat)).T/mu_hat).T

print("Distance between P and P_Mom_hat (computed using pi instead of the sample mean)")
print(error_computation(P_mom_hat))

