import os
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend (important!)
import matplotlib.pyplot as plt 
import numpy as np

# Folder to save plots
SAVE_DIR = os.path.expanduser("~/Downloads/pca_plots")
os.makedirs(SAVE_DIR, exist_ok=True)


def _save_plot(filename):
    path = os.path.join(SAVE_DIR, filename)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_original(X):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    plt.title("Original Data")
    _save_plot("original.png")


def plot_projection(Z):
    plt.figure()
    plt.scatter(Z, np.zeros_like(Z))
    plt.title("1D Projection")
    plt.yticks([])
    _save_plot("projection.png")


def plot_histogram(Z):
    plt.figure()
    plt.hist(Z, bins=20)
    plt.title("Distribution after Projection")
    _save_plot("histogram.png")


def plot_reconstruction(X, X_rec):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], label="Original")
    plt.scatter(X_rec[:, 0], X_rec[:, 1], label="Reconstructed")
    plt.legend()
    plt.title("Reconstruction Comparison")
    _save_plot("reconstruction.png")