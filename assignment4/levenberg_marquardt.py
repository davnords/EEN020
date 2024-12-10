import numpy as np
from tqdm import tqdm

def compute_J(P, X):
    c = P[2,:]@X
    J = np.array([
        (P[0, :]@X)/(c**2)*P[2,:]-P[0, :]/c,
        (P[1, :]@X)/(c**2)*P[2,:]-P[1, :]/c,
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


def compute_reprojection_error(P_1,P_2,X_j,x_1j,x_2j):
    r_1 = [x_1j[0] - (P_1[0, :]@X_j)/(P_1[2, :]@X_j), x_1j[1] - (P_1[1, :]@X_j)/(P_1[2, :]@X_j)]
    r_2 = [x_2j[0] - (P_2[0, :]@X_j)/(P_2[2, :]@X_j), x_2j[1] - (P_2[1, :]@X_j)/(P_2[2, :]@X_j)]
    
    r = np.array([r_1, r_2])
    err = np.linalg.norm(r)

    return err, r

def compute_total_reprojection_error(P_1,P_2,X,x):
    N = X.shape[-1]
    err = []
    for i in range(N):
        Xj = X[:, i]
        x1j = x[0, :, i]
        x2j = x[1, :, i]
        err += [compute_reprojection_error(P_1, P_2, Xj, x1j, x2j)[0]]
    return err

def compute_update(r, J, mu):
    C = J.T @ J + mu * np.eye(J.shape[1])
    c = J.T @ r
    delta = -np.linalg.solve(C, c)
    return delta

def perform_bundle_adjustment(X, x, P, epochs=5):
    X = X.copy()
    N = X.shape[-1]
    for _ in tqdm(range(epochs), desc='Performing bundle adjustment'):
        for i in range(N):
            x1j = x[0, :, i]
            x2j = x[1, :, i]

            r, J = linearize_reproj_err(P[0], P[1], X[:, i], x1j, x2j)
            delta_Xj = compute_update(r, J, 1)
            X[:, i] += delta_Xj
    return X