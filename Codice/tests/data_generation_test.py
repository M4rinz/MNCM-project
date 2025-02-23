"""This test is needed to test the generation of the data,
saving it into files, and loading them
"""

import os
import sys

# Launch this script from within the test/ folder!
sys.path.append(os.path.abspath(os.path.join('..')))

from utilities.data import generate_random_P, create_observations, return_subdir_name, save_observation
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

n_reps = 10

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

prod = product(T_range, K_range, range(0,len(alpha)), ['gaussian','binomial'])
for T, K, i, noise_type in tqdm(prod):
	# Get parameter, distinguishing the two cases
	parameter = np.sqrt(variance[i]) if noise_type=='gaussian' else alpha[i]
	## Create a dictionary to store the data, with parameters value
	#dict_entry = {
	#	'noise_type': noise_type,
	#	'TxK': T*K,
	#}
	#if noise_type == 'gaussian':
	#	dict_entry['stdev'] = parameter
	#else:
	#	dict_entry['alpha'] = parameter
	
	# Create subdirectory name according to a standard
	subdir_name = return_subdir_name(T=T, K=K, S=S, N=N,
								  noise_type=noise_type,
								  parameter=parameter)
	# Create the path of the subdirectory in which the arrays corresponding to
	# the given configurations of parameters have to be stored
	subdir_path = os.path.join(PATH,'observations',subdir_name)

	# Invariant: the noisy and original observations are either both present
	# or not present
	if not os.path.exists(subdir_path) or not os.listdir(subdir_path):
		if VERBOSE:
			print(f"The folder {subdir_path} doesn't exist or it exists but is empty, I'll create it and generate the data...")
		os.makedirs(subdir_path, exist_ok=True)

		# Create the n_reps observations to fill the folder right away
		for rep in range(n_reps):
			# Create name of the file in which to save the arrays
			n_filename = f"n_t_arr__repetition={rep}"
			y_filename = f"y_t_arr__repetition={rep}"
			par_name_val = "stdev" if noise_type=='gaussian' else 'alpha'
			A_filename = f"A_noise_type={noise_type}_{par_name_val}={parameter}"
		
			n_t_array, y_t_array, A = create_observations(
														T=T, K=K, N=N,
														P=P, pi_0=pi,
														noise_type=noise_type, 
														parameter=parameter
														)
			if VERBOSE:
				print(f"... and save it in {subdir_path}")

			save_observation(array=n_t_array,
							filename = n_filename,
							path=subdir_path)
			
			save_observation(array=y_t_array,
							filename = y_filename,
							path=subdir_path)
			
			save_observation(array=A,
							filename=A_filename,
							path=subdir_path)
	else:
		num_files = len(os.listdir(subdir_path))
		assert num_files == 2*n_reps+1, f"Error: in {subdir_name} there are {num_files} files, there should be 2*n_reps +1 = {2*n_reps+1}"
		if VERBOSE:
			print(f"The folder {subdir_path} already exists and is not empty, let's just load the data")
			print(f"n_reps = {n_reps}. There are {num_files} files in the directory {subdir_name}")
		
		# We're just testing the creation, no point in loading
		#n_t_array = load_observation(filename=n_filename, path=subdir_path)
		#y_t_array = load_observation(filename=y_filename, path=subdir_path)
		
	## method of moments estimator
	#P_mom, _, _ = P_mom_stationary
	

	
	
