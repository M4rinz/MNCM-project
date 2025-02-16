import numpy as np
#from typing import Iterable

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
        P = rng.rand(S,S)
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
    noise_models = {
        'gaussian': lambda n_t, rng, K, S, kwargs: (n_t + rng.normal(0, kwargs.get('stdev', kwargs.get('parameter')), size=(K,S)), np.eye(S)),
        'laplace': lambda n_t, rng, K, S, kwargs: (n_t + rng.laplace(0, kwargs.get('decay', kwargs.get('parameter')), size=(K,S)), np.eye(S)),
        'binomial': lambda n_t, rng, K, S, kwargs: (rng.binomial(n_t, kwargs.get('alpha', kwargs.get('parameter'))), kwargs.get('alpha', kwargs.get('parameter')) * np.eye(S)),
        'poisson': lambda n_t, rng, K, S, kwargs: (rng.poisson(n_t * kwargs.get('lambda', kwargs.get('parameter'))), kwargs.get('lambda', kwargs.get('parameter')) * np.eye(S))
    }

    if noise_type in noise_models:
        return noise_models[noise_type](n_t, rng, K, S, kwargs)
    else:
        verbose = kwargs.get('verbose', False)
        if verbose:
            print("keywords to add noise are: 'gaussian', 'poisson',\
                'laplace' and 'binomial'.") 
            print("No noise will be added,\
                the original input will be returned.")
        return (n_t, np.eye(S))


# TODO: fix this!!
def create_observations(T:int, K:int, N:int,
                        pi_0:np.ndarray,
                        stationary:bool,
                        *args,**kwargs) -> tuple[np.ndarray]:
    """Creates aggregate observations (noisy and not noisy)
    from a Markov chain with given parameters

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
    mu_t = pi_0.T
    n_t_vector, y_t_vector = [], []
    #A_t_vector = []
    for _ in range(T):
        # create K observations of the observed data 
        # (multinomial draw from the marginal distribution)
        n_t = np.random.multinomial(n=N, pvals=mu_t, size=K)
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

