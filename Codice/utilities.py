import numpy as np
from typing import Iterable

#def generate_random_y_t_array(K:int, N:int, S:int) -> np.ndarray:
#    '''
#    Generate K vectors of length S,
#    with non-negative elements summing to N.
#    '''
#    y_t_array = []
#    for _ in range(K):
#        vector = np.zeros(S)
#        vector[:S-1] = np.random.randint(0, N, S-1)
#        vector = vector / vector.sum() * N
#        vector = np.floor(vector)
#        vector[S-1] = N - vector[:S-1].sum()
#        vector = vector.astype(float)
#        y_t_array.append(vector)
#    return np.array(y_t_array)

def generate_random_P(S:int,*args,**kwargs) -> np.ndarray:
    '''
    Generate a random stochastic matrix of size S by S
    P = generate_random_P(S,'dirichlet',precision) to use Dirichlet 
    distributions, like the authors did.
    '''
    P = np.zeros(shape=(S,S))
    if args and (args[0] == 'dirichlet' or args[0] == 'Dirichlet'):
        # we parameterize the distribution like the authors did
        D = kwargs.get('precision', kwargs.get('parameter'))
        P = np.random.dirichlet(D*np.ones(S)/S,S)
    else:
        P = np.random.rand(S,S)
        for i in range(S):
            P[i,:] = P[i,:]/P[i,:].sum()
    return P

def add_noise(n_t:np.ndarray,
              noise_type:str=None,
              *args,**kwargs) -> tuple[np.ndarray]:
    """Adds noise to the observation. It adds noise to all the K trials (rows of n_t), independently

    Args:
        n_t (np.ndarray): K by S array of aggregated observations
        noise_type (str, optional): Type of noise to add (gaussian, poisson...). Defaults to None. If an invalid value is given, no noise will be added


    Returns:
        tuple[np.ndarray]: (y_t,A_t), i.e. noisy version of n_t and matrix of its expectation, given n_t
    """

    (K,S) = n_t.shape
    if noise_type == 'gaussian':
        sigma2 = kwargs.get('variance', kwargs.get('parameter'))
        return (n_t + np.random.normal(0,sigma2,size=(K,S)),np.eye(S))
    elif noise_type == 'laplace':
        lamda = kwargs.get('decay', kwargs.get('parameter'))
        return (n_t + np.random.laplace(0,lamda,size=(K,S)),np.eye(S))
    elif noise_type == 'binomial':
        alpha = kwargs.get('alpha', kwargs.get('parameter'))
        if isinstance(alpha,float) or isinstance(lamda,int):
            alpha *= np.ones(S)
        return (np.random.binomial(n_t,alpha),alpha*np.eye(S))
    elif noise_type == 'poisson':
        lamda = kwargs.get('lambda', kwargs.get('parameter'))    
        if isinstance(lamda,float) or isinstance(lamda,int):
            lamda *= np.ones(S)
        return (np.random.poisson(n_t*lamda),alpha*np.eye(S))
    else:
        verbose = kwargs.get('verbose',False)
        if verbose:
            print("keywords to add noise are: 'gaussian', 'poisson',\
                'laplace' and 'binomial'.") 
            print("No noise will be added,\
                the original input will be returned.")
        return (n_t,np.eye(S))

# Basically, this has to be refactored
def sanity_checks(K:int, N:int, S:int,
                  y_t_array:Iterable[np.ndarray],
                  A1:np.ndarray, A2:np.ndarray):
    # A1 and A2 must be S by S
    c1 = A1.shape == (S,S) and A2.shape == (S,S)
    # Each observation y_t must be a np.ndarray
    c2 = all([isinstance(x,np.ndarray) for x in y_t_array])
    # Each observation y_t must be a nonnegative vector summing to N
    c3 = all([(all(y_t >= 0) and y_t.sum() == N) for y_t in y_t_array])
    # y_t_array must consist of K observations of size S
    if isinstance(y_t_array,list) or isinstance(y_t_array,tuple):
        c4 = len(y_t_array) == K and all([y_t.shape == (S,) for y_t in y_t_array])
    elif isinstance(y_t_array,np.ndarray):
        c4 = y_t_array.shape == (K,S)
    else:
        # Exception??
        print('the array of observations must be either a tuple,\
              a list or a np.ndarray. I mean, please.')
    
    # Print alerts if any of the conditions are False
    if not c1:
        print("A1 and A2 aren't of size S by S")
    if not c2:
        print("There are non-np.ndarray observations")
    if not c3:
        print("There are non-nonnegative or non-summing-to-N observations")
    if not c4:
        print("The array of observations isn't of size K by S")

    # If all conditions are True, print a success message
    if c1 and c2 and c3 and c4:
        print("All sanity checks passed successfully.")
    

#TODO incorporate sanity checks
def P_mom_nonstationary(
        y_t_array:np.ndarray,
        y_tp1_array:np.ndarray,
        A_t:np.ndarray,
        A_tp1:np.ndarray,
        N:int) -> np.ndarray:
    """Computes the method of moments approximation of the transition
    matrix P of a Markov chain, using noisy aggregated data from
    specific timesteps t and t+1.

    Args:
        y_t_array (np.ndarray): K by S array with the noisy aggregated observations at time t, y_t
        y_tp1_array (np.ndarray): K by S array with the noisy aggregated observations at time t+1, y_{t+1}
        A_t (np.ndarray): S by S matrix of the conditional expectation of the original aggregated data n_t given y_t
        A_tp1 (np.ndarray): S by S matrix of the conditional expectation of the original aggregated data n_{t+1} given y_{t+1}
        N (int): number of individuals in the population

    Returns:
        np.ndarray: the estimation of the transition matrix
    """
    K,S = y_t_array.shape

    #TODO: improve this
    sanity_checks(K,N,S,y_t_array,A_t,A_tp1)

    # estimate mean of noisy data for timestep t
    # (aka compute the empirical expectation)
    m_t_hat = y_t_array.mean(axis=0)
    m_tp1_hat = y_tp1_array.mean(axis=0)

    # multiply by A_t^-1, as of proposition 3
    # (aka estimate mean of true counts)
    m_t_hat = np.linalg.solve(A_t,m_t_hat)
    m_tp1_hat = np.linalg.solve(A_tp1,m_tp1_hat)
    # Normalize
    mu_t_hat = m_t_hat/m_t_hat.sum()
    mu_tp1_hat = m_tp1_hat/m_tp1_hat.sum()

    # Estimate time-lagged covariance of true counts
    #Sigma_t_tp1_hat = sum((np.outer(y_t-mu_hat, y_t-mu_hat) for y_t in y_t_array))
        # Other formulation with broadcasting
    deviations_t = y_t_array - m_t_hat
    deviations_tp1 = y_tp1_array - m_tp1_hat
    Sigma_t_tp1_hat = np.matmul(deviations_t.T,deviations_tp1)/K
    # Correction with error matrices as of proposition 3
    # (i.e. Sigma_t_tp1_hat = A_t^-1*Sigma_t_tp1_hat*A_tp1^-1)
    Sigma_t_tp1_hat = np.linalg.solve(A_t,Sigma_t_tp1_hat)
    Sigma_t_tp1_hat = np.linalg.solve(A_tp1,Sigma_t_tp1_hat.T).T

    # Estimate transition matrix
    P_mom_t = ((Sigma_t_tp1_hat/N + np.outer(mu_t_hat,mu_tp1_hat)).T/mu_t_hat).T

    return P_mom_t


def P_mom_stationary(
        y_array:np.ndarray,
        A:np.ndarray,
        N:int) -> np.ndarray:
    """Computes the method of moments approximation of the transition
    matrix P of a Markov chain, using noisy aggregated data from
    all timesteps (strongly stationary case).

    Args:
        y_array (np.ndarray): array of shape (T,K,S) with the noisy aggregated observations
        A (np.ndarray): S by S matrix of the conditional expectation of the original aggregated data n given y
        N (int): number of individuals in the population

    Returns:
        np.ndarray: the estimation of the transition matrix
    """
    _, K, _ = y_array.shape

    #sanity_checks(K,N,S,)

    # estimate mean of noisy data
    # (aka compute empirical expectations along T and K)
    m_t_hat = y_array.mean(axis=(0,1))

    # estimate mean of true counts
    # (aka multiply by A^-1, as of proposition 3)
    m_t_hat = np.linalg.solve(A,m_t_hat)

    # Normalize
    mu_hat = m_t_hat/m_t_hat.sum()

    # estimate time-lagged covariance of noisy counts
    deviations = y_array - m_t_hat
    Sigma_hat = np.matmul(deviations[:-1].transpose(0,2,1),deviations[1:]).mean(axis=0)/K
    # estimate time-lagged covariance of true counts
    Sigma_hat = np.linalg.solve(A,Sigma_hat)
    Sigma_hat = np.linalg.solve(A,Sigma_hat.T).T

    # Estimate transition matrix
    P_mom = ((Sigma_hat/N+np.outer(mu_hat,mu_hat)).T/mu_hat).T

    return P_mom

