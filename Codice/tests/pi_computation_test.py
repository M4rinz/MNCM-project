"""
This test is just to check that the numerical methods to compute the 
invariant distribution of a stochastic matrix work as expected 
"""

import os
import sys

# Launch this script from within the test/ folder!
sys.path.append(os.path.abspath(os.path.join('..')))

from utilities.data import generate_random_P
from utilities.num_methods import compute_stationary_LU_GTH, compute_stationary_LU_rough

import numpy as np

def print_help():
	print("Usage: python pi_computation_test.py [S] [VERBOSE] [SEED]")
	print("Arguments:")
	print("  S        : Size of the matrix (default: 10)")
	print("  VERBOSE  : Verbose output (default: True)")
	print("  SEED     : Random seed (default: 42)")
	print("  --help   : Show this help message")

# Default parameters
S = 10
VERBOSE = True
SEED = 42

# Parse command-line arguments
if '--help' in sys.argv:
	print_help()
	sys.exit(0)

if len(sys.argv) > 1:
	S = int(sys.argv[1])
if len(sys.argv) > 2:
	VERBOSE = sys.argv[2].lower() in ['true', '1', 't', 'y', 'yes']
if len(sys.argv) > 3:
	SEED = int(sys.argv[3])

P = generate_random_P(S, seed=SEED)
A = np.eye(S) - P

pi_rough, _, _ = compute_stationary_LU_rough(P)
pi_gth = compute_stationary_LU_GTH(P)

res = pi_rough @ A
if VERBOSE:
	print('(I-P.T) @ pi, with pi computed "roughly" (i.e. LU factorization performed by SciPy)')
	print(res)
	print()
assert np.isclose(res, np.finfo(np.dtype(res[0])).eps).all(), "Error: the relative error is greater than machine precision"

res = pi_gth @ A
if VERBOSE:
	print("pi.T @ (I-P), with pi computed by solving pi.T @ L = e_n, where L comes from the LU factorization with diagonal adjustments")
	print(res)
	print()
assert np.isclose(res, np.finfo(np.dtype(res[0])).eps).all(), "Error: the relative error is greater than machine precision"

print("All tests passed successfully!")