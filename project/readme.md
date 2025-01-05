# Project SfM David Nordström

## Get started

Run 

```bash
python3 run_sfm --dataset x
```

Where `x` corresponds to the dataset to which you want to apply the SfM pipeline. 


## OpenCV
If you want to speed things up you can run stuff with OpenCV instead. Here are some useful commands

```python
# Essential matrix
E, mask = cv2.findEssentialMat(
    x1n[:2].T, x2n[:2].T, 
    cameraMatrix=np.eye(3),  # Identity since points are normalized
    method=cv2.RANSAC,
    prob=0.999,
    threshold=eps
) 

_, R, t, _ = cv2.recoverPose(E, x1n[:2].T, x2n[:2].T)  

# PnP 
rvec, _ = cv2.Rodrigues(R)
success, rvec, T, inliers = cv2.solvePnPRansac(objectPoints.T, imagePoints, K, None, rvec=rvec, useExtrinsicGuess=True)
R_new, _ = cv2.Rodrigues(rvec)

```