from utils import generate_data
from lda import LDA
from visualize import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Generate Synthetic data 
X, y = generate_data(n_classes=2,samples_per_class=100,n_features = 2,class_separation=4,within_class_variance=1,random_state=42)


plot_original(X,y)

# Plotting Original data
lda = LDA(n_components=1)


lda.fit(X, y)

Z = lda.transform(X)

plot_projected_data(Z, y)
