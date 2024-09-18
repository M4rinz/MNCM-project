import numpy as np

def generate_y_t_array(K:int, N:int, S:int) -> np.ndarray:
    '''
    Generate K vectors of length S,
    with non-negative elements summing to N.
    '''
    y_t_array = []
    for _ in range(K):
        vector = np.zeros(S)
        vector[:S-1] = np.random.randint(0, N, S-1)
        vector = vector / vector.sum() * N
        vector = np.floor(vector)
        vector[S-1] = N - vector[:S-1].sum()
        vector = vector.astype(float)
        y_t_array.append(vector)
    return np.array(y_t_array)

def sanity_checks(K:int, N:int, S:int,
                  y_t_array,
                  A1, A2):
    # A1 and A2 must be S by S
    c1 = A1.shape == (S,S) and A2.shape == (S,S)
    # Each observation y_t must be a np.ndarray
    c2 = all([isinstance(x,np.ndarray) for x in y_t_array])
    # Each observation y_t must be a nonnegative vector summing to N
    c3 = all([(y_t > 0 and y_t.sum() == N) for y_t in y_t_array])
    # y_t_array must consist of K observations of size S
    if isinstance(y_t_array,list) or isinstance(y_t_array,tuple):
        c4 = len(y_t_array) == K and y_t_array[0].shape == (S,)
    elif isinstance(y_t_array,np.ndarray):
        c4 = y_t_array.shape == (K,S)
    else:
        # Exception??
        print('the array of observations must be either a tuple,\
              a list or a np.ndarray. I mean, please.')
        
    print(c1*c2*c3*c4*'There are the following mistakes')
    print(c1*"A1 and A2 aren't of size S by S")
    print(c2*"There are non-np.ndarray observations")
    print(c3*"There are non-nonnegative or non-summing-to-N observations")
    print(c4*"The array of observations isn't of size K by S")
    

def P_mom_nonstationary(
        y_t_array,
        A_t,
        A_tp1,
        N:int):
    K = len(y_t_array)
    S = A_t.shape[0]

    # estimate mean of noisy data for the current timestep
    # (aka compute the empirical expectation)
    m_t_hat = y_t_array.mean(axis=0)

    # multiply by A_t^-1, as of proposition 3
    # (aka estimate mean of true counts)
    m_t_hat = np.linalg.solve(A_t,m_t_hat)
    # Normalize
    mu_hat = m_t_hat/m_t_hat.sum()

    # Estimate time-lagged covariance of true counts
    #Sigma_hat = sum((np.outer(y_t-mu_hat, y_t-mu_hat) for y_t in y_t_array))
        # Other formulation
    deviations = y_t_array - mu_hat
    Sigma_hat = np.dot(deviations.T,deviations)/K
    # Correction with error matrices as of proposition 3
    Sigma_hat = np.linalg.solve(A_t,Sigma_hat)
    Sigma_hat = np.linalg.solve(A_tp1,Sigma_hat.T).T

    # Estimate transition matrix
    P_mom_t = ((Sigma_hat/N + np.outer(mu_hat,mu_hat)).T/mu_hat).T

    return P_mom_t




