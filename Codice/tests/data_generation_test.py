"""This test is needed to test the generation of the data,
saving it into files, and loading them
"""

import os
import sys

# Launch this script from within the test/ folder!
sys.path.append(os.path.abspath(os.path.join('..')))

from utilities.data import generate_random_P, create_observations, load_observation, save_observation, return_filename
from utilities.num_methods import compute_stationary_LU_GTH

import numpy as np
from itertools import product
from tqdm import tqdm


# Default parameters
S = 10
VERBOSE = True
SEED = 42
SUBDIRECTORY = 'test_folder'
PATH = os.path.join('..','data',SUBDIRECTORY)

# grid of parameters values
alpha = [1, 0.5]
variance = [0, 1]

T_range = [10**k for k in range(1,3)]
K_range = [1, 5, 20]
N = 100

n_reps = 100

P = generate_random_P(S, seed=SEED)
pi = compute_stationary_LU_GTH(P)

# to save P
with open(os.path.join(PATH,'P.npy'), 'wb') as file:
	np.save(file, P)

#to load P
with open(os.path.join(PATH,'P.npy'), 'rb') as file:
	P2 = np.load(file)

assert np.array_equal(P,P2), "Error: original matrix P doesn't equal the saved and loaded version"
if VERBOSE:
	print("Saving and loading P was successful")

del P2 # let's save space

with open(os.path.join(PATH,'pi.npy'), 'wb') as file:
	np.save(file, pi)


#sys.exit(0)	# stop here
#print("Non arrivare qui!")

prod = product(T_range, K_range, range(0,len(alpha)))
for T, K, i in tqdm(prod):
	# Create the filename according to a standard
	n_filename = return_filename(noisy=False,
							  	T=T, K=K, S=S, N=N,
							  	noise_type='gaussian',
							  	parameter=np.sqrt(variance[i])
							  	)
	y_filename = return_filename(noisy=True,
							  T=T, K=K, S=S, N=N,
							  noise_type='gaussian',
							  parameter=np.sqrt(variance[i]))
	
	# For the original observations
	if os.path.exists(os.path.join(PATH, n_filename)):
		if VERBOSE:
			print("The file already exists, let's just load it")
		n_t_array_gauss = load_observation(filename=n_filename, path=PATH)
	else:
		if VERBOSE:
			print("The file doesn't exist, I'll create it...")
		n_t_array_gauss, y_t_array_gauss, A_gauss = create_observations(
														T=T, K=K, N=N,
														P=P, pi_0=pi,
														noise_type='gaussian', 
														stdev=np.sqrt(variance[i]) 
													)
		if VERBOSE:
			print(f"... and save it in {PATH}")
		save_observation(array=n_t_array_gauss,
						filename = n_filename,
						path=PATH
						)
		
		save_observation(array=y_t_array_gauss,
						filename = y_filename,
						path=PATH
						)
	

	
	
