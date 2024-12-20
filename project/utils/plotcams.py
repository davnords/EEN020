import numpy as np
import matplotlib.pyplot as plt

def plotcams(P, ax, scale=1.5):
    c = np.zeros((4, len(P)))
    v = np.zeros((3, len(P)))
    
    for i in range(len(P)):
        c[:, i] = np.linalg.svd(P[i])[2][-1]  
        v[:, i] = P[i][2, :3]  
    
    c /= c[3, :]
    
    ax.quiver(c[0, :], c[1, :], c[2, :], v[0, :], v[1, :], v[2, :],
              color='r', linewidth=1.5, length=scale, normalize=True)
    
    
