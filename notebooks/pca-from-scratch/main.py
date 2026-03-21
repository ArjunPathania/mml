from utils import generate_data
from pca import PCA_FromScratch
from visualize import *
from sklearn.decomposition import PCA

# Step 1: Generate data
X = generate_data(n_samples=200)

plot_original(X)

# Step 2: PCA from scratch
pca = PCA_FromScratch(n_components=1)
pca.fit(X)

Z = pca.transform(X)
X_rec = pca.inverse_transform(Z)

plot_projection(Z)
plot_histogram(Z)
plot_reconstruction(X, X_rec)

# Step 3: Compare with sklearn
sk_pca = PCA(n_components=1)
Z_sklearn = sk_pca.fit_transform(X)

print("Manual PCA (first 5):", Z[:5])
print("Sklearn PCA (first 5):", Z_sklearn[:5])
