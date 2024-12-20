import numpy as np

def homography_to_RT(H, x1, x2):
    def unitize(a, b):
        denom = 1.0 / np.sqrt(a**2 + b**2)
        return a * denom, b * denom

    # Ensure H has the correct sign
    N = x1.shape[1]
    if x1.shape[0] != 3:
        x1 = np.vstack((x1, np.ones((1, N))))
    if x2.shape[0] != 3:
        x2 = np.vstack((x2, np.ones((1, N))))
    
    positives = np.sum(np.sum(x2 * (H @ x1), axis=0) > 0)
    if positives < N / 2:
        H *= -1

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    
    s1 = S[0] / S[1]
    s3 = S[2] / S[1]
    zeta = s1 - s3

    a1 = np.sqrt(1 - s3**2)
    b1 = np.sqrt(s1**2 - 1)
    a, b = unitize(a1, b1)
    c, d = unitize(1 + s1 * s3, a1 * b1)
    e, f = unitize(-b / s1, -a / s3)

    v1 = V[:, 0]
    v3 = V[:, 2]

    n1 = b * v1 - a * v3
    n2 = b * v1 + a * v3

    R1 = U @ np.array([[c, 0, d], [0, 1, 0], [-d, 0, c]]) @ V.T
    R2 = U @ np.array([[c, 0, -d], [0, 1, 0], [d, 0, c]]) @ V.T

    t1 = e * v1 + f * v3
    t2 = e * v1 - f * v3

    if n1[2] < 0:
        t1 = -t1
        n1 = -n1
    if n2[2] < 0:
        t2 = -t2
        n2 = -n2

    # Convert to H&Z notation
    t1 = R1 @ t1
    t2 = R2 @ t2

    return R1, t1, R2, t2
