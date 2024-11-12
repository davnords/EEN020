import matplotlib.pyplot as plt
from pflat import pflat

def project_and_plot(P, Xs, image, save_path, title='3D projection into camera P', show=True):
    x = pflat(P @ Xs)
    plt.figure()
    plt.imshow(image.convert("RGB"))
    plt.plot(x[0], x[1], 'ro', alpha=0.3, markersize=0.7)
    plt.title(title)
    plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()