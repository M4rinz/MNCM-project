import numpy as np
from utilities.data import generate_random_P
from utilities.num_methods import compute_stationary_LU_GTH, compute_stationary_LU_rough

S = 10
P = generate_random_P(S)
A = np.eye(S)-P
pi_rough, _, _ = compute_stationary_LU_rough(P)
print(pi_rough@A)
pi_gth = compute_stationary_LU_GTH(P)
print(pi_gth@A)