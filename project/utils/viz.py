from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
from matplotlib.cm import get_cmap
from .plotcams import plotcams

def plot_scene(Xs, Ps, title='Scene reconstruction SfM', name='scene_reconstruction.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(Xs[:, 0], Xs[:, 1], Xs[:, 2], 'bo', alpha=0.2, markersize=0.5)
    plotcams(Ps, ax=ax, scale=0.5)
    plt.axis('equal')
    plt.title(title)
    plt.savefig(f"plots/{name}")
    plt.show()

def plot_colored_scene(Xs, Ps, title='Scene reconstruction SfM', name='scene_reconstruction.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plotcams(Ps, ax=ax, scale=0.5)

    cmap = get_cmap('tab10')  # You can use other colormaps like 'viridis', 'plasma', etc.


    num_colors = len(Xs)
    for i, X in enumerate(Xs):
        color = cmap(i / num_colors)  # Normalize the index to the range [0, 1]
        ax.plot(X[:, 0], X[:, 1], X[:, 2], 'o', alpha=0.2, markersize=0.5, color=color)

    plt.axis('equal')
    plt.title(title)
    plt.savefig(f"plots/{name}")
    plt.show()
