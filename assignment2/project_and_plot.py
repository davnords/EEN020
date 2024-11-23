import matplotlib.pyplot as plt
from pflat import pflat

def project_and_plot(P, Xs, image, save_path, title='3D projection into camera P', show=True, other_points=[]):
    x = pflat(P @ Xs)
    plt.figure()
    plt.imshow(image.convert("RGB"))
    plt.plot(x[0], x[1], 'ro', alpha=0.3, markersize=0.7, label='Projected points')
    for points in other_points:
        plt.plot(points[0], points[1], 'o', markeredgecolor='b', markerfacecolor='none', alpha=0.5, markersize=0.5, markeredgewidth=0.5, label='Other points')
    plt.legend(fontsize=10, markerscale=10)
    plt.title(title)
    plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()