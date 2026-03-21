import numpy as np

def generate_data(n_samples=200, noise=0.5, random_state=42):
    np.random.seed(random_state)

    # Create correlated 2D data
    x = np.random.randn(n_samples)
    y = 2 * x + noise * np.random.randn(n_samples)

    X = np.column_stack((x, y))
    return X