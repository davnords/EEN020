from matplotlib import pyplot as plt
from .plotcams import plotcams

def plot_scene(Xs, Ps, title='Scene reconstruction SfM', name='scene_reconstruction.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(Xs[:, 0], Xs[:, 1], Xs[:, 2], 'bo', alpha=0.2, markersize=0.5)
    plotcams(Ps, ax=ax, scale=0.1)
    plt.axis('equal')
    plt.title(title)
    plt.savefig(f"plots/{name}")
    plt.show()
    plt.close()