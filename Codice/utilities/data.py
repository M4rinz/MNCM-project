import numpy as np
#from typing import Iterable
from itertools import product

def generate_random_P(S:int,*args,**kwargs) -> np.ndarray:
    '''
    Generate a random stochastic matrix of size S by S\n
    P = generate_random_P(S,'dirichlet',precision) to use Dirichlet 
    distributions, like the authors did.
    '''
    rng = kwargs.get('rng', np.random.default_rng(kwargs.get('seed')))
    if rng is None:
        rng = np.random.default_rng()
    P = np.zeros(shape=(S,S))
    if args and args[0].lower() == 'dirichlet':
        # we parameterize the distribution like the authors did
        D = kwargs.get('precision', kwargs.get('parameter'))
        P = rng.dirichlet(D*np.ones(S)/S,S)
    else:
        P = rng.random(size=(S,S))
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
    
    rng = kwargs.get('rng', np.random.default_rng(kwargs.get('seed')))
    if rng is None:
        rng = np.random.default_rng()
    (K,S) = n_t.shape
    #noise_models = {
    #    'gaussian': lambda n_t, rng, K, S, kwargs: (n_t + rng.normal(0, kwargs.get('stdev', kwargs.get('parameter')), size=(K,S)), np.eye(S)),
    #    'laplace': lambda n_t, rng, K, S, kwargs: (n_t + rng.laplace(0, kwargs.get('decay', kwargs.get('parameter')), size=(K,S)), np.eye(S)),
    #    'binomial': lambda n_t, rng, K, S, kwargs: (rng.binomial(np.int(n_t), kwargs.get('alpha', kwargs.get('parameter'))), kwargs.get('alpha', kwargs.get('parameter')) * np.eye(S)),
    #    'poisson': lambda n_t, rng, K, S, kwargs: (rng.poisson(n_t * kwargs.get('lambda', kwargs.get('parameter'))), kwargs.get('lambda', kwargs.get('parameter')) * np.eye(S))
    #}
#
    #if noise_type in noise_models:
    #    return noise_models[noise_type](n_t, rng, K, S, kwargs)
    #else:
    #    verbose = kwargs.get('verbose', False)
    #    if verbose:
    #        print("keywords to add noise are: 'gaussian', 'poisson',\
    #            'laplace' and 'binomial'.") 
    #        print("No noise will be added,\
    #            the original input will be returned.")
    #    return (n_t, np.eye(S))

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


# DON'T USE THIS!!!
def create_observations_dummy(T:int, K:int, N:int,
                        pi_0:np.ndarray,
                        stationary:bool,
                        *args,**kwargs) -> tuple[np.ndarray]:
    """Creates aggregate observations (noisy and not noisy)
    from a Markov chain with given parameters, by repeatedly 
    sampling from the distribution of the states

    Args:
        T (int): n° of timesteps
        K (int): n° of repeated observations
        N (int): population size
        pi_0 (np.ndarray): initial distribution
        stationary (bool): if the Markov process is strictly stationary
        (i.e. if pi_0 is the steady-state vector)

    Returns:
        tuple[np.ndarray]: (true aggregate data, noisy data, noise model matrix)
    """
    rng = kwargs.get('rng', np.random.default_rng(kwargs.get('seed')))
    if rng is None:
        rng = np.random.default_rng()

    mu_t = pi_0.T
    n_t_vector, y_t_vector = [], []
    #A_t_vector = []
    for _ in range(T):
        # create K observations of the observed data 
        # (multinomial draw from the marginal distribution)
        n_t = rng.multinomial(n=N, pvals=mu_t, size=K)
        # create noisy observations 
        # all observations will have the same noise type
        y_t, A_t = add_noise(n_t, *args, **kwargs)
        # append the observations
        n_t_vector.append(n_t)
        y_t_vector.append(y_t)
        # In the stationary case, the marginal distribution is equal 
        # to the stationary distribution
        if not stationary:
            # update the distribution of x_t for the next iteration
            P = kwargs.get('P')
            mu_t = np.dot(mu_t,P)
    # Simplified case (same as the article):
    # Noise model is fixed and known in advance. Parameters of the noise models are the same across time
    # Therefore A_t is the same for all timesteps 
    return (np.array(n_t_vector), np.array(y_t_vector), A_t)    


def create_observations(T:int, K:int, N:int,
                        P:np.ndarray,
                        pi_0:np.ndarray,
                        *args, **kwargs) -> tuple[np.ndarray]:
    """Creates aggregated observations (noisy and not noisy)
    from a Markov Chain by sampling the initial distribution from pi_0,
    and then repeatedly sampling from each row i of the transition matrix
    a distribution for a number of people proportional to how many are in state i
    (i.e. sample from multinomial(n_t(i),P[i,:])).
    The process is repeated K times.
    Noise is added independetly (just pass the right parameters)

    Args:
        T (int): n° of timesteps
        K (int): n° of repeated observations
        N (int): population size
        P (np.ndarray): transition matrix
        pi_0 (np.ndarray): initial distribution

    Returns:
        tuple[np.ndarray]: (true aggregate data, noisy data, noise model matrix)
    """
    rng = kwargs.get('rng', np.random.default_rng(kwargs.get('seed')))
    if rng is None:
        rng = np.random.default_rng()

    S = P.shape[0]
    noise_type = kwargs.get('noise_type')
    # To create noisy observations with the binomial distribution int type is needed 
    n_t_vector = np.zeros((T,K,S), dtype='int64')   
    y_t_vector = np.zeros((T,K,S), dtype='int64' if noise_type in ['binomial'] else 'float64')
    A_t_vector = np.zeros((T,S,S))

    # Initial population is a multinomial draw from the initial distribution
    n_0 = rng.multinomial(n=N, pvals=pi_0, size=K)
    y_t_vector[0,:,:], A_t_vector[0,:,:] = add_noise(n_0, *args, **kwargs)

    n_t_vector[0,:,:] = n_0

    for t in range(T-1):
        n_tp1 = np.zeros((K,S), dtype='int64')
        # P[i,:] encodes the probability of going from state i to another states. 
        # Thus, to simulate where the n_t(i) individuals that are in state i at time t
        # go in the next timestep t+1, we sample from a multinomial distribution
        # with n=n_t(i) (population size), and pvals=P[i,:] (parameter of the multinomial)
        for k, i in product(range(K), range(S)):
            # The distribution of the population at time t+1 is obtained by gathering
            # (i.e. summing), for all i, where the n_t(i) individuals went
            n_tp1[k,:] += np.random.multinomial(n_t_vector[t,k,i], P[i,:])

        y_tp1, A_tp1 = add_noise(n_tp1, *args, **kwargs)
        
        n_t_vector[t+1,:,:] = n_tp1
        y_t_vector[t+1,:,:] = y_tp1
        A_t_vector[t+1,:,:] = A_tp1

    # Simplified case (same as the article):
    # Noise model is fixed and known in advance. Parameters of the noise models are the same across time
    # Therefore A_t is the same for all timesteps.
    # Thus, even though we've gathered all matrices so far, we only return the first :)

    return n_t_vector, y_t_vector, A_t_vector[0,:,:] 
