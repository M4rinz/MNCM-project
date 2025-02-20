"""
This test is needed to try out the simulation method of the Markov Chain
(i.e. simulate the evolution of the population)
"""

import os
import sys

# Launch this script from within the test/ folder!
sys.path.append(os.path.abspath(os.path.join('..')))

from utilities.data import add_noise, generate_random_P, create_observations
from utilities.num_methods import compute_stationary_LU_GTH

import numpy as np
from itertools import product

def internal_add_noise(n_t:np.ndarray,
              			noise_type:str=None,
              			*args,**kwargs) -> tuple[np.ndarray]:
	"""Adds noise to the observation. It adds noise to all the K trials (rows of n_t), independently

	Args:
		n_t (np.ndarray): K by S array of aggregated observations
		noise_type (str, optional): Type of noise to add (gaussian, poisson...). Defaults to None. If an invalid value is given, no noise will be added

	Returns:
		tuple[np.ndarray]: (y_t,A_t), i.e. noisy version of n_t and matrix of its expectation, given n_t
	"""

	rng = kwargs.get('rng', np.random.default_rng(kwargs.get('seed')))
	if rng is None:
		rng = np.random.default_rng()
	(K,S) = n_t.shape

	if noise_type == 'gaussian':
		sigma = kwargs.get('stdev', kwargs.get('parameter'))
		return (n_t + rng.normal(0, sigma, size=(K, S)), np.eye(S))
	elif noise_type == 'laplace':
		lamda = kwargs.get('decay', kwargs.get('parameter'))
		return (n_t + rng.laplace(0, lamda, size=(K, S)), np.eye(S))
	elif noise_type == 'binomial':
		alpha = kwargs.get('alpha', kwargs.get('parameter'))
		return (rng.binomial(np.int64(n_t), alpha), alpha * np.eye(S))
	elif noise_type == 'poisson':
		lamda = kwargs.get('lambda', kwargs.get('parameter'))
		return (rng.poisson(n_t * lamda), lamda * np.eye(S))
	else:
		verbose = kwargs.get('verbose', False)
		if verbose:
			print("keywords to add noise are: 'gaussian', 'poisson', 'laplace' and 'binomial'.")
			print("No noise will be added, the original input will be returned.")
		return (n_t, np.eye(S))


# Default parameters
S = 10
VERBOSE = True
SEED = 42


# Parameters of the test
T = 1000
K = 50
N = 100

# Change here to change the variance
variance = 0.7
noise_type = 'binomial'

P = generate_random_P(S, seed=SEED)
pi = compute_stationary_LU_GTH(P)

n_t_vector = np.zeros((T,K,S))
y_t_vector = np.zeros((T,K,S))
A_t_vector = np.zeros((T,S,S))

n_0 = np.random.multinomial(n=N, pvals=pi, size=K)
y_t_vector[0,:,:], A_t_vector[0,:,:] = internal_add_noise(n_0, seed=SEED,
					 							noise_type=noise_type, 
												parameter=np.sqrt(variance))

n_t_vector[0,:,:] = n_0

for t in range(T-1):
	n_tp1 = np.zeros((K, S))
	for k, i in product(range(K), range(S)):
		n_tp1[k,:] += np.random.multinomial(n_t_vector[t,k,i], P[i,:])

	y_tp1, A_tp1 = internal_add_noise(n_tp1, seed=SEED,
							 		noise_type=noise_type, 
									parameter=np.sqrt(variance))

	n_t_vector[t+1,:,:] = n_tp1
	y_t_vector[t+1,:,:] = y_tp1

if noise_type == 'binomial':
	assert (np.unique(y_t_vector - np.int64(y_t_vector)) == 0).all(), "Error: y_t_vector has non-integer values"
	if VERBOSE:
		print(f"y_t_vector.dtype = {y_t_vector.dtype}, but it only has integer values")
	#assert n_t_vector.dtype == np.float64 and y_t_vector.dtype == np.int64, f"Error: array of (noisy) observations is of type {y_t_vector.dtype} instead of int64"
	#if VERBOSE:
	#	print(f"The array of (noisy) observations is of the correct type (i.e. {y_t_vector.dtype})")
else:
	assert y_t_vector == n_t_vector.dtype == np.float64, f"Error:  n_t_vector.dtype = {n_t_vector.dtype}, y_t_vector.dtype = {y_t_vector.dtype} instead of both being of type float64."
	if VERBOSE:
		print("The array of observations are both of the correct type (i.e. float64)")

assert n_t_vector.shape == (T,K,S), f"Error: array of observations is of shape {n_t_vector.shape} instead of ({T},{K},{S})"
if VERBOSE:
	print(f"The array of observations is correctly shaped (i.e. it has shape ({T},{K},{S}))")

assert np.all(n_t_vector.sum(axis=2) == N), f"Error: not all observations have total count of N = {N}"
if VERBOSE:
	print(f"All observations sum to the population size {N}")