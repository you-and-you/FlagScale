import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


def visualize_3d(one_latents, another_latents):
    n_steps_one = one_latents.shape[0]
    n_steps_another = another_latents.shape[0]

    latents_flat_one = one_latents.reshape(n_steps_one, -1)
    pca_one = PCA(n_components=3)
    latents_3d_one = pca_one.fit_transform(latents_flat_one)

    latents_flat_another = another_latents.reshape(n_steps_another, -1)
    pca_another = PCA(n_components=3)
    latents_3d_another = pca_another.fit_transform(latents_flat_another)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter_one = ax.scatter(
        latents_3d_one[:, 0],
        latents_3d_one[:, 1],
        latents_3d_one[:, 2],
        c=range(n_steps_one),
        cmap='viridis',
        alpha=0.7,
        s=50,
        label='one_latents',
    )

    for i in range(n_steps_one - 1):
        ax.plot(
            [latents_3d_one[i, 0], latents_3d_one[i + 1, 0]],
            [latents_3d_one[i, 1], latents_3d_one[i + 1, 1]],
            [latents_3d_one[i, 2], latents_3d_one[i + 1, 2]],
            'blue',
            alpha=0.3,
        )

    scatter_another = ax.scatter(
        latents_3d_another[:, 0],
        latents_3d_another[:, 1],
        latents_3d_another[:, 2],
        c=range(n_steps_another),
        cmap='plasma',
        alpha=0.7,
        s=50,
        label='another_latents',
    )

    for i in range(n_steps_another - 1):
        ax.plot(
            [latents_3d_another[i, 0], latents_3d_another[i + 1, 0]],
            [latents_3d_another[i, 1], latents_3d_another[i + 1, 1]],
            [latents_3d_another[i, 2], latents_3d_another[i + 1, 2]],
            'red',
            alpha=0.3,
        )

    ax.scatter(
        latents_3d_one[0, 0],
        latents_3d_one[0, 1],
        latents_3d_one[0, 2],
        c='green',
        s=100,
        marker='o',
        label='Start (one_latents)',
    )
    ax.scatter(
        latents_3d_one[-1, 0],
        latents_3d_one[-1, 1],
        latents_3d_one[-1, 2],
        c='blue',
        s=100,
        marker='s',
        label='End (one_latents)',
    )

    ax.scatter(
        latents_3d_another[0, 0],
        latents_3d_another[0, 1],
        latents_3d_another[0, 2],
        c='yellow',
        s=100,
        marker='^',
        label='Start (another_latents)',
    )
    ax.scatter(
        latents_3d_another[-1, 0],
        latents_3d_another[-1, 1],
        latents_3d_another[-1, 2],
        c='red',
        s=100,
        marker='d',
        label='End (another_latents)',
    )

    cbar_one = plt.colorbar(scatter_one, ax=ax, pad=0.1)
    cbar_one.set_label('20 Steps Diffusion Progress')
    cbar_another = plt.colorbar(scatter_another, ax=ax, pad=0.15)
    cbar_another.set_label('50 Steps Diffusion Progress')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D PCA Visualization of Two Diffusion Processes (one vs another latents)')

    ax.legend()

    plt.savefig("3d_pca_visualization_one_vs_another_latents.png", dpi=300, bbox_inches='tight')
    plt.savefig("3d_pca_visualization_one_vs_another_latents.pdf", bbox_inches='tight')

    plt.show()


def main():
    one_latents = np.load("latents_20_steps.npy")
    another_latents = np.load("latents_50_steps.npy")
    visualize_3d(one_latents, another_latents)


if __name__ == "__main__":
    main()
