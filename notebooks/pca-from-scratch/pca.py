import numpy as np

class PCA_FromScratch:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        # Step 1: Center data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Step 2: Covariance matrix
        cov = np.cov(X_centered, rowvar=False)

        # Step 3: Eigen decomposition
        eigen_vals, eigen_vecs = np.linalg.eig(cov)

        # Step 4: Sort eigenvalues
        idx = np.argsort(eigen_vals)[::-1]
        eigen_vals = eigen_vals[idx]
        eigen_vecs = eigen_vecs[:, idx]

        # Step 5: Select components
        self.components = eigen_vecs[:, :self.n_components]
        self.explained_variance = eigen_vals

        return self

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components

    def inverse_transform(self, Z):
        return Z @ self.components.T + self.mean