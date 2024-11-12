import numpy as np

def camera_center_and_axis(p):
    """Calculates the camera center and the (normalized) principal axis using the camera matrix."""
    if p.shape != (3, 4):
        raise ValueError('Input must be a 3x4 matrix')
    
    _, _, Vt = np.linalg.svd(p)
    C = Vt[-1]
    C = C[:-1] / C[-1]

    Q = p[:3, :3]
    
    axis = Q[:, 0:3].T @ np.array([0, 0, 1])

    axis = axis / np.linalg.norm(axis)

    assert np.abs(np.linalg.norm(axis)-1)<1e-4, 'Principal axis is not unit vector' 

    return C, axis
