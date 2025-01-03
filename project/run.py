from utils.get_dataset_info import get_dataset_info
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from utils.get_dataset_info import get_dataset_info
import cv2 as cv
import numpy as np
from utils import general, pflat, estimate_T_robust, levenberg_marquardt, triangulate_3D_point_DLT, viz
from romatch import roma_outdoor
import torch

K, img_names, init_pair, pixel_threshold = get_dataset_info(1)
K_inv = np.linalg.inv(K)
image_paths = ['./project_data/'+img_name for img_name in img_names]

# Normalize pixel threshold
focal_length = (K[0,0] + K[1 ,1])/2 # Average over the diagonal
epipolar_treshold =          pixel_threshold / focal_length
homography_threshold =   3 * pixel_threshold / focal_length
translation_threshold  = 3 * pixel_threshold / focal_length

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
roma_model = roma_outdoor(device=device)

relative_rotations = general.find_relative_rotation(image_paths, K_inv, roma_model, epipolar_treshold, device=device)
absolute_rotations = general.upgrade_to_absolute_rotation(relative_rotations)

imA_path = image_paths[init_pair[0]-1]
Ps = []
for i in range(len(image_paths)):
    R = absolute_rotations[i]
    if i == init_pair[0]-1:
        print('Skipping...')
        T=np.zeros((3,1))
        Ps.append(np.hstack((R, T)))
    else:
        x1u, x2u = general.find_matches(imA_path, image_paths[i], roma_model, device=device)
        x1n = pflat.pflat(K_inv @ x1u)
        x2n = pflat.pflat(K_inv @ x2u)
        X = general.triangulate_initial_points(x1n, x2n, epipolar_treshold)
        X = (absolute_rotations[init_pair[0]-1]).T@X.T
        Ps.append(estimate_T_robust.estimate_T_robust(x2n, general.make_homogenous(X.T), R, translation_threshold, iterations=10))

viz.plot_scene(X, Ps, image_paths)