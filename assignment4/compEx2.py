from scipy.io import loadmat
from PIL import Image
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
from estimate_E_robust import estimate_E_robust
from pflat import pflat
from extract_P_from_E import extract_P_from_E
from triangulate_3D_point_DLT import triangulate_3D_point_DLT
from plotcams import plotcams

if __name__ == "__main__":
    mat = loadmat('./A4data/compEx2data.mat')
    K = mat['K']
    K_inv = np.linalg.inv(K)

    images = [Image.open('./A4data/fountain1.png'), Image.open('./A4data/fountain2.png')] 
    im1 = cv2.imread('./A4data/fountain1.png') 
    im2 = cv2.imread('./A4data/fountain2.png') 
    gray_im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray_im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(gray_im1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_im2, None)

    print('Found %d keypoints in image 1' % len(keypoints1))
    print('Found %d keypoints in image 2' % len(keypoints2))

    img1 = cv2.drawKeypoints(gray_im1, keypoints1, im1)
    cv2.imwrite('plots/compEx2_plot1.png', img1)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    print('Found %d matches' % len(matches))

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # 0.75 is the ratio
            good_matches.append(m)

    print('Found %d good matches' % len(good_matches))

    x1 = np.array([[keypoints1[match.queryIdx].pt[0], keypoints1[match.queryIdx].pt[1]] for match in good_matches])
    x2 = np.array([[keypoints2[match.trainIdx].pt[0], keypoints2[match.trainIdx].pt[1]] for match in good_matches])

    x1 = np.vstack((x1.T, np.ones(x1.shape[0])))
    x2 = np.vstack((x2.T, np.ones(x2.shape[0])))
    
    x = np.array([x1, x2])

    x1_normalized = pflat(np.dot(K_inv, x1))
    x2_normalized = pflat(np.dot(K_inv, x2))
    x_normalized = np.array([x1_normalized, x2_normalized])

    # Finding the essential matrix
    inlier_threshold_px = 2
    eps = inlier_threshold_px * 2 / ( K[0,0] + K[1 ,1])
    E, _ = estimate_E_robust(x1_normalized, x2_normalized, eps, iterations=10000)

    P1 = np.hstack((np.eye(3), np.zeros((3,1))))
    P2s = extract_P_from_E(E)

    N = x1.shape[-1]
    camera_counter = 1

    for P2 in P2s:
        Xj = []
        positive_depth_count = 0
        for i in range(N):
            xi = x_normalized[:, :, i]
            Xi, _ = triangulate_3D_point_DLT(xi[0, :2], xi[1, :2], P1, P2)
            Xj.append(Xi)

            X_h = np.hstack((Xi, [1]))  # Homogeneous coordinates
            if P1[2, :] @ X_h > 0 and P2[2, :] @ X_h > 0:
                positive_depth_count += 1

        print(f"Camera {camera_counter} has {positive_depth_count/N*100:.2f}% points in front of both cameras")
        Xj = np.array(Xj)

        # Compute distances from the origin
        distances = np.linalg.norm(Xj, axis=1)

        # Find the 90th percentile distance
        threshold = np.percentile(distances, 90)

        # Filter out points beyond the threshold
        Xj_filtered = Xj[distances <= threshold]

        # Plot the filtered points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(Xj_filtered[:, 0], Xj_filtered[:, 1], Xj_filtered[:, 2], 'bo', alpha=0.2, markersize=0.5)
        plotcams([P1, P2], ax=ax, scale=0.1)
        plt.axis('equal')
        plt.title(f"Camera {camera_counter} 3D points (filtered) and cameras")
        plt.savefig(f"plots/compEx2_plot{2+camera_counter}_filtered.png")
        plt.show()
        plt.close()

        camera_counter += 1

