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

    bf = cv2.BFMatcher(cv2.NORM_L2)  # L2 norm for SIFT
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    print('Found %d matches' % len(matches))

    x1 = np.array([[keypoints1[match.queryIdx].pt[0], keypoints1[match.queryIdx].pt[1]] for match in matches])
    x2 = np.array([[keypoints2[match.trainIdx].pt[0], keypoints2[match.trainIdx].pt[1]] for match in matches])
    
    # x1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches])  # Points from image A
    # x2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches])  # Points from image B

    x1 = np.vstack((x1.T, np.ones(x1.shape[0])))
    x2 = np.vstack((x2.T, np.ones(x2.shape[0])))
    
    x = np.array([x1, x2])

    x1_normalized = pflat(np.dot(K_inv, x1))
    x2_normalized = pflat(np.dot(K_inv, x2))
    x_normalized = np.array([x1_normalized, x2_normalized])

    im1 = cv2.imread('./A4data/fountain1.png') 
    im2 = cv2.imread('./A4data/fountain2.png') 

    matches = random.sample(matches, 10)

    matches_img = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches[:500], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.title('Randomly selected 100 matches')
    plt.imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
    plt.savefig('plots/compEx2_plot2.png')
    # plt.show()
    plt.close()

    # Finding the essential matrix
    inlier_threshold_px = 2
    eps = inlier_threshold_px * 2 / ( K[0,0] + K[1 ,1])
    E, _ = estimate_E_robust(x1_normalized, x2_normalized, eps, iterations=1000)


    P1 = np.hstack((np.eye(3), np.zeros((3,1))))
    P2s = extract_P_from_E(E)

    N = x1.shape[-1]

    camera_counter = 0

    for P2 in P2s:
        Xj = []
        positive_depth_count = 0
        for i in range(N):
            xi = x[:, :, i]
            Xi, _ = triangulate_3D_point_DLT(xi[0, :2], xi[1, :2], P1, P2)
            Xj.append(Xi)

            X_h = np.hstack((Xi, [1]))  # Homogeneous coordinates
            if P1[2, :] @ X_h > 0 and P2[2, :] @ X_h > 0:
                positive_depth_count += 1

        Xj = np.array(Xj)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(Xj[:, 0], Xj[:, 1], Xj[:, 2], 'bo', alpha=0.2, markersize=0.5)
        plotcams([P1, P2], ax=ax, scale=3)
        plt.axis('equal')
        plt.title('3D points and cameras')
        plt.savefig(f"plots/compEx2_plot{3+camera_counter}.png")
        plt.show()
        plt.close()

        camera_counter += 1
