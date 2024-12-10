import numpy as np

def compute_reprojection_error(P_1,P_2,X_j,x_1j,x_2j):
    r_1 = [x_1j[0] - (P_1[0, :]@X_j)/(P_1[2, :]@X_j), x_1j[1] - (P_1[1, :]@X_j)/(P_1[2, :]@X_j)]
    r_2 = [x_2j[0] - (P_2[0, :]@X_j)/(P_2[2, :]@X_j), x_2j[1] - (P_2[1, :]@X_j)/(P_2[2, :]@X_j)]

    r = np.array([r_1.T, r_2.T])
    err = np.linalg.norm(r)

    return err, r