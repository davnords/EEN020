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
    mu = 0.1  # Initial damping factor

    for epoch in tqdm(range(epochs), desc='Performing bundle adjustment'):
        total_error = 0

        for i in range(N):
            x1j = x[0, :, i]
            x2j = x[1, :, i]

            # Compute reprojection error and Jacobian
            r, J = linearize_reproj_err(P[0], P[1], X[:, i], x1j, x2j)
            
            # Compute initial error for current point
            current_error = np.linalg.norm(r)**2

            # Compute update for X[:, i]
            delta_Xj = compute_update(r, J, mu)

            # Temporarily update the solution
            X_temp = X[:, i] + delta_Xj

            # Recompute error after applying the update
            r_new, _ = linearize_reproj_err(P[0], P[1], X_temp, x1j, x2j)
            new_error = np.linalg.norm(r_new)**2

            if new_error < current_error:
                # Accept the update and reduce the damping factor
                X[:, i] = X_temp
                mu = max(mu / 10, 1e-7)  # Ensure mu doesn't become too small
            else:
                # Reject the update and increase the damping factor
                mu = min(mu * 10, 1e7)  # Ensure mu doesn't become too large

            total_error += new_error

        # Optionally, log or print the total error for monitoring
        print(f"Epoch {epoch + 1}/{epochs}, Total Error: {total_error:.6f}, Damping Factor: {mu:.6e}")

    return X