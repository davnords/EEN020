import random
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2

if __name__=="__main__":
    sift = cv2.SIFT_create(contrastThreshold=0.01) 

    images = [Image.open('A2data/data/cube1.JPG'), Image.open('A2data/data/cube2.JPG')] 
    im1 = cv2.imread('A2data/data/cube1.JPG') 
    im2 = cv2.imread('A2data/data/cube2.JPG') 
    gray_im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray_im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    keypoints1, descriptors1 = sift.detectAndCompute(gray_im1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_im2, None)
    f1 = np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle] for kp in keypoints1]).T
    f2 = np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle] for kp in keypoints2]).T

    img1 = cv2.drawKeypoints(gray_im1, keypoints1, im1)
    cv2.imwrite('plots/compEx3_plot1.png', img1)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    x1 = np.array([[keypoints1[match.queryIdx].pt[0], keypoints1[match.queryIdx].pt[1]] for match in matches])
    x2 = np.array([[keypoints2[match.trainIdx].pt[0], keypoints2[match.trainIdx].pt[1]] for match in matches])

    np.save('variables/x1_compEx3.npy', x1)
    np.save('variables/x2_compEx3.npy', x2)

    matches = random.sample(matches, 10)

    im1 = cv2.imread('A2data/data/cube1.JPG') 
    im2 = cv2.imread('A2data/data/cube2.JPG') 
    matches_img = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches[:500], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.title('Randomly selected 100 matches')
    plt.imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
    plt.savefig('plots/compEx3_plot2.png')
    plt.show()
    plt.close()


