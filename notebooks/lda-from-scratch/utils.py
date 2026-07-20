# Generate sythetic data set for LDA

# LDA assumptions
# 1. Gaussian Distribution
# 2. Equal Covariance Matrix
# 3. Independent observations

import numpy as np

def generate_data(n_classes:int = 2,
                  samples_per_class:int = 1000,
                  n_features:int=2,
                  class_separation:int=5,
                  within_class_variance:int = 1,
                  random_state:int = 42):

    # For reproduceable results
    rng = np.random.default_rng(random_state)


    # data set 
    X = []  # Feature matrix
    y = []  # Class Labels 

    covariance = within_class_variance * np.eye(n_features) # Shared covariance matrix for all classes (LDA assumption)

    for cls in range(n_classes):
        
        # Random class mean
        mean = rng.uniform(
            low = cls * class_separation,
            high = (cls + 1) * class_separation,
            size = n_features,
        )

        # Generate samples
        samples = rng.multivariate_normal(
            mean = mean,
            cov = covariance,
            size = samples_per_class
        )

        X.append(samples)
        y.extend([cls] * samples_per_class)

    X = np.vstack(X)
    y = np.array(y)

    return X, y

# Future possible additions 
# shuffle = True ,to randomly mix the samples before returning them
# return_means=False , to optionally return the true class means for visualization and debugging.