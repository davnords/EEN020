from utils.get_dataset_info import get_dataset_info
from PIL import Image
import cv2 as cv
import numpy as np
from utils import general, pflat, estimate_E_robust
from matplotlib import pyplot as plt
import random

def sfm(dataset):
    K, img_names, init_pair, pixel_threshold = get_dataset_info(dataset)
    
    K_inv = np.linalg.inv(K)

    # Load images
    pil_images = [Image.open('./project_data/'+img_name) for img_name in img_names]
    cv_images = [cv.cvtColor(np.array(pil_image), cv.COLOR_BGR2GRAY) for pil_image in pil_images]

    sift = cv.SIFT_create()
    bf = cv.BFMatcher()

    # Pairwise matching followed by RANSAC
    for i in range(len(cv_images)-1):
        im1 = cv_images[i]
        im2 = cv_images[i+1]

        keypoints1, descriptors1 = sift.detectAndCompute(im1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(im2, None)

        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good_matches = []
        for m, n in matches:
            if m.distance < 0.25 * n.distance:  # 0.75 is the ratio
                good_matches.append(m)

        print('Number of good matches:', len(good_matches))

        x1u = general.make_homogenous( np.array([[keypoints1[match.queryIdx].pt[0], keypoints1[match.queryIdx].pt[1]] for match in good_matches]) )
        x2u = general.make_homogenous( np.array([[keypoints2[match.queryIdx].pt[0], keypoints2[match.queryIdx].pt[1]] for match in good_matches]) )

        x1n = pflat.pflat(K_inv @ x1u)
        x2n = pflat.pflat(K_inv @ x2u)

        # Step 2: Compute the Essential Matrix
        eps = pixel_threshold * 2 / ( K[0,0] + K[1 ,1])

        # Find the essential matrix
        E, inliers = estimate_E_robust.estimate_E_robust(x1n, x2n, eps, iterations=10)
        

        # img3 = cv.drawMatches(im1,keypoints1,im2,keypoints2,good_matches[indices],None,**draw_params)
        # plt.imshow(img3, 'gray'),plt.show()