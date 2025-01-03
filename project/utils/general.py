from tqdm import tqdm
import numpy as np
from .parallell_RANSAC import parallell_RANSAC
from .triangulate_3D_point_DLT import triangulate_3D_point_DLT
from .extract_P_from_E import extract_P_from_E
from .estimate_E_robust import estimate_E_robust
from .levenberg_marquardt import perform_bundle_adjustment
from .pflat import pflat
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


def find_relative_rotation(image_paths, K_inv, model, eps, device='cuda:0'):
    out = []
    for i in range(len(image_paths)-1):
        imA_path = image_paths[i]
        imB_path = image_paths[i+1]
        x1u, x2u = find_matches(imA_path, imB_path, model, device=device)
        x1n = pflat(K_inv @ x1u)
        x2n = pflat(K_inv @ x2u)
        R = parallell_RANSAC(x1n, x2n, eps, iterations=100)[:3, :3]
        out.append(R)
    return out


def upgrade_to_absolute_rotation(relative_rotations):
    absolute_rotation = []
    for i in range(len(relative_rotations)+1):
        if i == 0:
            absolute_rotation.append(np.eye(3))
        else: 
            absolute_rotation.append(relative_rotations[i-1]@absolute_rotation[i-1])
    return np.array(absolute_rotation)

def triangulate_initial_points(x1n, x2n, eps):
    E, _ = estimate_E_robust(x1n, x2n, eps, iterations=100)

    P1 = np.hstack((np.eye(3), np.zeros((3,1))))
    P2s = extract_P_from_E(E)

    xn = np.array([x1n, x2n])

    N = x1n.shape[-1]
    
    Xjs = []
    depth_counts = []
    for P2 in tqdm(P2s, desc='Triangulating points...'):
        Xj = []
        positive_depth_count = 0
        for i in range(N):
            xi = xn[:, :, i]
            Xi, _ = triangulate_3D_point_DLT(xi[0, :2], xi[1, :2], P1, P2)
            Xj.append(Xi)

            X_h = np.hstack((Xi, [1]))  # Homogeneous coordinates
            if P1[2, :] @ X_h > 0 and P2[2, :] @ X_h > 0:
                positive_depth_count += 1
        Xj = np.array(Xj)
        Xjs.append(Xj)
        depth_counts.append(positive_depth_count)
    
    depth_counts = np.array(depth_counts)
    highest_depth_count = np.argmax(depth_counts)
    return Xjs[highest_depth_count]

def remove_3D_outliers(X, percentile=90):
    # Compute distances from the origin
    distances = np.linalg.norm(X, axis=1)

    # Find the 90th percentile distance
    threshold = np.percentile(distances, percentile)

    indices= distances <= threshold

    # Filter out points beyond the threshold
    X = X[indices]
    return X, indices