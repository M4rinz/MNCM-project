import numpy as np

def generate_y_t_vector(K:int, N:int, S:int) -> np.ndarray:
    '''
    Generate K vectors of length S,
    with non-negative elements summing to N.
    '''
    y_t_vector = []
    for _ in range(K):
        vector = np.zeros(S)
        vector[:S-1] = np.random.randint(0, N, S-1)
        vector = vector / vector.sum() * N
        vector = np.floor(vector)
        vector[S-1] = N - vector[:S-1].sum()
        vector = vector.astype(float)
        y_t_vector.append(vector)
    return np.array(y_t_vector)



# def P_mom_nonstationary(
#         y_t_vector,
#         A_t,
#         A_tp1,
#         N:int):
#     K = len(y_t_vector)
