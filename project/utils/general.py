import numpy as np
from .parallell_RANSAC import parallell_RANSAC
from .triangulate_3D_point_DLT import triangulate_3D_point_DLT
from .extract_P_from_E import extract_P_from_E
from .estimate_E_robust import estimate_E_robust
from .levenberg_marquardt import perform_bundle_adjustment
from PIL import Image

def make_homogenous(x):
    return np.vstack((x.T, np.ones(x.shape[0])))

def find_matches(imA_path, imB_path, model, device='cuda:0'):
    imA = Image.open(imA_path) 
    imB = Image.open(imB_path) 

    imA_dims = (imA.height, imA.width)
    imB_dims = (imB.height, imB.width)

    H_A, W_A = imA_dims
    H_B, W_B = imB_dims
    
    warp, certainty = model.match(imA_path, imB_path, device=device)
    matches, certainty = model.sample(warp, certainty)
    kptsA, kptsB = model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
    kptsA, kptsB = kptsA.cpu().numpy(), kptsB.cpu().numpy()

    return make_homogenous(kptsA), make_homogenous(kptsB)


def find_relative_rotation_and_translation(x1n, x2n, eps):
    return parallell_RANSAC(x1n, x2n, eps, iterations=10)

def upgrade_to_absolute_rotation(relative_rotations):
    absolute_rotation = []
    for i in range(len(relative_rotations)+1):
        if i == 0:
            absolute_rotation.append(np.eye(3))
        else: 
            absolute_rotation.append(relative_rotations[i-1]@absolute_rotation[i-1])
    return np.array(absolute_rotation)

def triangulate_initial_points(x1n, x2n, eps):
    E, _ = estimate_E_robust(x1n, x2n, eps, iterations=10)

    P1 = np.hstack((np.eye(3), np.zeros((3,1))))
    P2s = extract_P_from_E(E)

    xn = np.array([x1n, x2n])

    N = x1n.shape[-1]
    camera_counter = 1

    for P2 in P2s:
        Xj = []
        positive_depth_count = 0
        for i in range(N):
            xi = xn[:, :, i]
            Xi, _ = triangulate_3D_point_DLT(xi[0, :2], xi[1, :2], P1, P2)
            Xj.append(Xi)

            X_h = np.hstack((Xi, [1]))  # Homogeneous coordinates
            if P1[2, :] @ X_h > 0 and P2[2, :] @ X_h > 0:
                positive_depth_count += 1

        print(f"Camera {camera_counter} has {positive_depth_count/N*100:.2f}% points in front of both cameras")
        Xj = np.array(Xj)

        # Compute distances from the origin
        # distances = np.linalg.norm(Xj, axis=1)

        # Find the 90th percentile distance
        # threshold = np.percentile(distances, 90)

        # Filter out points beyond the threshold
        # Xj_filtered = Xj[distances <= threshold]

    return Xj
