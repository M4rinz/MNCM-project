import numpy as np


# ################################# COMPUTATION OF P_MOM #################################   

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
    K,_ = y_t_array.shape

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

    #sanity_checks_stationary(K,N,S,)

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



# ################################# COMPUTATION OF P_CLS #################################
    
def P_cls_stationary(y_array:np.ndarray,
                     rcond:float=-1):
    """Computes the Conditional Least Squares approximation of the
    transition matrix P or a Markov chain, using noisy aggregated 
    data from all timesteps (strongly stationary case).
    Since the data comes from multiple repetitions, an average is taken. 

    The function uses `np.linalg.lstsq` to solve the linear 
    least square problem that appears. 

    Args:
        y_array (np.ndarray): array of shape (T,K,S) with the noisy aggregated observations
        rcond (float, optional): rcond parameter for NumPy. Defaults to -1.

    Returns:
        _type_: _description_
    """
    _, K, _ = y_array.shape
    # "flip" dimensions T and K 
    X = y_array[:-1].transpose(1,0,2)
    Y = y_array[1:].transpose(1,0,2)
    # We take an average over the K trials
    coll_P_cls = np.array([np.linalg.lstsq(X[k], Y[k],rcond=rcond)[0] for k in range(K)])
    P_cls = coll_P_cls.mean(axis=0)

    return P_cls         