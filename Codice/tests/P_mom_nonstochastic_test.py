"""This test has a case in which the method of moments estimator is 
not a stochastic matrix: it's got negative elements, 
and its rows don't sum to one
"""

import os
import sys

# Launch this script from within the test/ folder!
sys.path.append(os.path.abspath(os.path.join('..')))

from utilities.data import generate_random_P, create_observations
from utilities.num_methods import compute_stationary_LU_GTH
from utilities.estimators import P_mom_stationary

import numpy as np

# Parameters 
S = 5
SEED = 42
VERBOSE = True

# Parameters related to observations
T = 20
K = 5
N = 25


P = generate_random_P(S, seed=SEED)
pi = compute_stationary_LU_GTH(P)

# Change here to change the variance
variance = 5

if VERBOSE:
	print(f"S = {S}, variance = {variance}")
	print(f"T = {T}, K = {K}, N = {N}")
	print()

n_t_array, y_t_array, A = create_observations(T=T, K=K, N=N,
											  P=P, pi_0=pi,
											  noise_type='gaussian', 
											  stdev=np.sqrt(variance))

# Are there negative observations?
lista_negs = [np.any(y_t_array[i,:,:] < 0) for i in range(T)]
if VERBOSE and np.any(lista_negs):
	print("There are negative elements in the noisy observations")
	if np.all(lista_negs):
		print("In particular, all timesteps have negative observations")
	else:
		print(f"In particular, timesteps {[x[0] for x in np.argwhere(lista_negs)]} have negative observations")
	print()

P_mom, mu, Sigma = P_mom_stationary(y_array=y_t_array, N=N, A=A)

negs_in_P = np.any(P<0)
if VERBOSE and negs_in_P:
	print("There are negative elements in P")
	print()

if VERBOSE:
	print(f"P_mom @ e = {P_mom.sum(axis=1)}")
	print()
	

