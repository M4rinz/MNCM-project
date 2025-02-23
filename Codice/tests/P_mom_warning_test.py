"""This test is the warning about division by zero when computing the estimator
"""

import os
import sys
import warnings

# Launch this script from within the test/ folder!
sys.path.append(os.path.abspath(os.path.join('..')))

from utilities.estimators import P_mom_stationary
from utilities.data import generate_random_P, create_observations
from utilities.num_methods import compute_stationary_LU_GTH

import numpy as np

# Default parameters
S = 10
D = 0.5
VERBOSE = True
SEED = 42

P = generate_random_P(S, 'dirichlet', precision=D, seed=SEED)
pi = compute_stationary_LU_GTH(P=P)


# Let's reproduce an "unfavourable setting"
T = 10	# n° of total timesteps
K = 1	# n° of total repetitions
N = 100


# Change here to change the variance
variance = 0


n_t_array, y_t_array, A = create_observations(T=T, K=K, N=N, 
											  P=P, pi_0=pi, 
											  stationary=True,
											  noise_type='gaussian', stdev=np.sqrt(variance), seed=SEED)

x = "hello"
#with np.errstate(divide='warn'):
with warnings.catch_warnings(record=True) as w:
	warnings.simplefilter("always")  # Ensure all warnings are caught
	P_mom, _, _ = P_mom_stationary(y_array=y_t_array,A=A, N=N)
	if w:
		x = "I caught the warning"
		if VERBOSE:
			print(x)

if np.any(np.isnan(P_mom)):
	assert x == "I caught the warning", "Error: warning not taught"
