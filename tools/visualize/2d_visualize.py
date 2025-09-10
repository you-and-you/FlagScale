import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusers import StableDiffusionPipeline
from PIL import Image
from sklearn.decomposition import PCA


def visualize_2d(one_latents, another_latents):
    n_steps_one = one_latents.shape[0]
    n_steps_another = another_latents.shape[0]
    latents_flat_one = one_latents.reshape(n_steps_one, -1)
    latents_flat_another = another_latents.reshape(n_steps_another, -1)

    pca = PCA(n_components=2)

    latents_2d_one = pca.fit_transform(latents_flat_one)
    latents_2d_another = pca.fit_transform(latents_flat_another)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        latents_2d_one[:, 0],
        latents_2d_one[:, 1],
        alpha=0.7,
        c=range(n_steps_one),
        cmap='Blues',
        marker='o',
        label='latents_2d_one',
    )
    plt.scatter(
        latents_2d_another[:, 0],
        latents_2d_another[:, 1],
        alpha=0.7,
        c=range(n_steps_another),
        cmap='Reds',
        marker='s',
        label='latents_2d_another',
    )
    plt.xlabel('Principal Component 1 (PC1)')
    plt.ylabel('Principal Component 2 (PC2)')
    plt.title('2D PCA Visualization of Diffusion Process Latents')
    plt.colorbar(label='Diffusion Step')
    plt.grid(True, alpha=0.3)

    for i, (x, y) in enumerate(latents_2d_one):
        if i % 2 == 0:
            plt.annotate(
                str(i),
                (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                color='blue',
                weight='bold',
            )

    for i, (x, y) in enumerate(latents_2d_another):
        if i % 2 == 0:
            plt.annotate(
                str(i),
                (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                color='red',
                weight='bold',
            )

    plt.savefig("pca_visualization_2d.png", dpi=300, bbox_inches='tight')
    plt.savefig("pca_visualization_2d.pdf", bbox_inches='tight')

    plt.show()


def main():
    one_latents = np.load("latents_20_steps.npy")
    another_latents = np.load("latents_50_steps.npy")
    visualize_2d(one_latents, another_latents)


if __name__ == "__main__":
    main()
