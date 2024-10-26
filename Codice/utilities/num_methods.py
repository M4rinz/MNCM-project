import numpy as np
import scipy

def compute_stationary_LU_rough(P:np.ndarray) -> tuple[np.ndarray, float, float]:
    #let's start very roughly: solve (I-P.T)@pi = 0 via LU factorization
    S = P.shape[0]
    A = np.eye(S)-P.T
    M,L,U = scipy.linalg.lu(A)
    LU_error = np.linalg.norm(M@L@U - A)
    pi = np.zeros(S)
    pi[-1] = 1
    for i in range(S-2,-1,-1):
        pi[i] = (-U[i,i+1:]@pi[i+1:])/U[i,i]
    pi /= pi.sum()
    res = np.linalg.norm(A@pi)
    return pi, res, LU_error


def compute_stationary_LU_GTH(P:np.ndarray) -> np.ndarray:
    S = P.shape[0]
    A = np.eye(S)-P
    # diagonal adjustment
    for i in range(S):
        A[i,i] = -A[i,:i].sum()-A[i,i+1:].sum()
    # Gaussia elimination
    for k in range(S-1):
        # Update L
        A[k+1:,k] /= A[k,k] 
        # Update U
        for i in range(k+1,S):
            A[i,k+1:] -= A[i,k]*A[k,k+1:]
        # Diagonal adjustment
        for i in range(k+1,S):
            A[i,i] = -A[i,k+1:i].sum()-A[i,i+1:].sum()
        
    # Back substitution
    pi = np.ones(S)
    for j in range(S-2,-1,-1):
        s = 0
        for i in range(j+1,S):
            s += pi[i]*A[i,j]
        pi[j] = -s

    return pi/pi.sum()