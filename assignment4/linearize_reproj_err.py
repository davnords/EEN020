import numpy as np

def compute_J(P, X):
    c = P[2,:]@X
    J = np.array([
        P[0, :]/(c**2)*P[2,:]-P[0, :]/c,
        P[1, :]/(c**2)*P[2,:]-P[1, :]/c,
    ])
    return J

def compute_R(P, X, x):
    c = P[2,:]@X
    R = np.array([
       x[0]- P[0, :]@X/c,
       x[1]- P[1, :]@X/c,
    ])
    return R

def linearize_reproj_err(P_1,P_2,X_j,x_1j,x_2j):
    r = np.concatenate([compute_R(P_1, X_j, x_1j), compute_R(P_2, X_j, x_2j)])
    J = np.concatenate([compute_J(P_1, X_j), compute_J(P_2, X_j)])
    return r, J
