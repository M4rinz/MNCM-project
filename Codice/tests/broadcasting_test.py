"""The purpose of this test is to check whether for subtracting the mean from 
each observation, broadcasting 
(i.e. doing y_t_array - m_hat, which means subtracting two arrays of different
shapes and rely on NumPy magic to sort things out)
is the same as doing it "manually" in a for loop.
"""
import os
import sys

# Launch this script from within the test/ folder!
sys.path.append(os.path.abspath(os.path.join('..')))

from utilities.data import generate_random_P, create_observations
from utilities.num_methods import compute_stationary_LU_GTH
from utilities.estimators import P_mom_stationary

from itertools import product

import numpy as np

def print_help():
	print("Usage: python broadcasting_test.py [S] [SEED] [VERBOSE]")
	print("Arguments:")
	print("  S       : Number of states (default: 10)")
	print("  SEED    : Random seed (default: 42)")
	print("  VERBOSE : Verbose output (default: True)")

# Default parameters
S = 10
SEED = 42
VERBOSE = True

# Parse command-line arguments
if '--help' in sys.argv:
	print_help()
	sys.exit(0)

if len(sys.argv) > 1:
	S = int(sys.argv[1])
if len(sys.argv) > 2:
	SEED = int(sys.argv[2])
if len(sys.argv) > 3:
	VERBOSE = sys.argv[3].lower() in ['true', '1', 't', 'y', 'yes']

P = generate_random_P(S)
pi = compute_stationary_LU_GTH(P=P)
if VERBOSE:
	print("pi =")
	print(pi)
	print()

# Parameters
T = 100
K = 2
N = 100

# Change here to change the variance (variance = 0 implies n_t_array == y_t_array)
variance = 1

# generate data
n_t_array, y_t_array, A = create_observations(T=T, K=K, N=N, 
											  P = P, pi_0=pi, 
											  stationary=True,
											  noise_type='gaussian', stdev=np.sqrt(variance), seed=SEED)
assert n_t_array.shape == (T,K,S), f"Error: the shape of the observation is {n_t_array.shape} instead of ({T},{K},{S})"
if VERBOSE:
	print("The shape of the observation is correct")
	print()

m_hat = y_t_array.mean(axis=(0,1))
m_hat = np.linalg.solve(A, m_hat)

# Compute y_t^{(k)} - m_hat using broadcasting
deviations = y_t_array - m_hat

# Compute y_t^{(k)} - m_hat
deviations_manual = np.zeros_like(deviations)
for t, k in product(range(T), range(K)):
	deviations_manual[t,k,:] = y_t_array[t,k,:] - m_hat

assert np.isclose(deviations, deviations_manual).all(), "Error: manual computation and broadcasting differ"
if VERBOSE:
	print("Manual computation roughly equals broadcasting")
	print()

P_mom, _, Sigma_hat = P_mom_stationary(y_t_array, A, N)
P_mom_manual = np.zeros_like(P_mom)

mu_hat = m_hat/np.linalg.norm(m_hat, 1)

# Check if mu_hat is normalized
assert np.isclose(np.linalg.norm(mu_hat, 1), 1), "Error: The L1 norm of mu_hat is not equal to 1."
if VERBOSE:    
	print(f"norm(mu_hat, 1) = {np.linalg.norm(mu_hat, 1)}")
	print("mu_hat is correctly normalized")
	print()

Sigma_hat_manual = np.matmul(deviations_manual[:-1].transpose(0,2,1),deviations_manual[1:]).mean(axis=0)/K
Sigma_hat_manual = np.linalg.solve(A,Sigma_hat_manual)
Sigma_hat_manual = np.linalg.solve(A,Sigma_hat_manual.T).T

# Check if doing the product by hand is the same as using np.outer
outer_manual = np.matmul(mu_hat.reshape((mu_hat.shape[0],1)),mu_hat.reshape((1,mu_hat.shape[0])))
assert np.array_equal(outer_manual, np.outer(mu_hat,mu_hat)), "Error: manual computation and np.outer differ"
if VERBOSE:
	print("np.outer and mu_hat @ mu_hat.T yield the same result")
	print()

P_mom_manual = ((Sigma_hat_manual/N+np.outer(mu_hat,mu_hat)).T/mu_hat).T

assert np.isclose(P_mom, P_mom_manual, equal_nan=True).all(), "Error: using broadcasting yields a different estimator than the manual computation"
if VERBOSE:
	print("Using broadcasting and using manual computations return the same estimator")
	print()

print("All tests passed successfully!")
