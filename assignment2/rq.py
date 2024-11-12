import numpy as np

def rq(a):
    """
    Perform RQ factorization on matrix 'a' such that a = RQ,
    where R is an upper triangular matrix and Q is a unitary matrix.
    
    Parameters:
        a (np.ndarray): The input matrix to be factorized.
    
    Returns:
        r (np.ndarray): The upper triangular matrix R.
        q (np.ndarray): The unitary matrix Q.
    """
    m, n = a.shape
    e = np.eye(m)
    p = e[:, ::-1]  # reverse the order of columns to create permutation matrix
    q0, r0 = np.linalg.qr((p @ a[:, :m].T) @ p)

    r = (p @ r0.T) @ p
    q = (p @ q0.T) @ p

    # Ensure R has positive diagonal elements by adjusting with a fix matrix
    fix = np.diag(np.sign(np.diag(r)))
    r = r @ fix
    q = fix @ q

    # If a is not square, augment Q
    if n > m:
        q = np.hstack((q, np.linalg.inv(r) @ a[:, m:n]))

    return r, q
