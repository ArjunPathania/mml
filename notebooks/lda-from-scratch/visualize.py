import os
import numpy as np
import matplotlib.pyplot as plt

# Directory to save plots
SAVE_DIR = os.path.expanduser("~/Downloads/lda_plots")
os.makedirs(SAVE_DIR, exist_ok=True)


def _save_plot(filename):

    path = os.path.join(SAVE_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_original(X, y):
 

    plt.figure(figsize=(6, 6))

    for cls in np.unique(y):
        X_cls = X[y == cls]
        plt.scatter(
            X_cls[:, 0],
            X_cls[:, 1],
            s=40,
            alpha=0.8,
            label=f"Class {cls}"
        )

    plt.title("Original Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")

    _save_plot("original.png")


def plot_projected_data(Z, y=None):

    Z = np.asarray(Z)

    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    n_components = Z.shape[1]

    plt.figure(figsize=(6, 6))



    if n_components == 1:

        if y is None:
            plt.scatter(Z[:, 0], np.zeros(len(Z)), s=40)
        else:
            for cls in np.unique(y):
                Z_cls = Z[y == cls]
                plt.scatter(
                    Z_cls[:, 0],
                    np.zeros(len(Z_cls)),
                    s=40,
                    label=f"Class {cls}"
                )

            plt.legend()

        plt.xlabel("LD1")
        plt.yticks([])
        plt.title("LDA Projection (1D)")



    elif n_components == 2:

        if y is None:
            plt.scatter(Z[:, 0], Z[:, 1], s=40)
        else:
            for cls in np.unique(y):
                Z_cls = Z[y == cls]
                plt.scatter(
                    Z_cls[:, 0],
                    Z_cls[:, 1],
                    s=40,
                    label=f"Class {cls}"
                )

            plt.legend()

        plt.xlabel("LD1")
        plt.ylabel("LD2")
        plt.title("LDA Projection (2D)")
        plt.axis("equal")

    else:
        raise ValueError(
            f"Visualization only supports 1 or 2 components, got {n_components}."
        )

    plt.grid(True, alpha=0.3)

    _save_plot("projection.png")