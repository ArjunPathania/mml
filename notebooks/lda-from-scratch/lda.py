import numpy as np

class LDA:

    def __init__(self, n_components=None):
        """
        Parameters
        ----------
        n_components : int
            Number of linear discriminants to keep.
        """
        # Hyperparameter
        self.n_components = n_components

        # Learned parameters
        self.classes = None
        self.class_means = None
        self.overall_mean = None
        self.n_features = None
        self.components = None



    def calculate_overall_mean(self, X):
        """
        Compute the overall mean vector.

        Returns
        -------
        overall_mean : ndarray of shape (n_features,)
        """
        self.overall_mean = np.mean(X, axis=0)
        return self.overall_mean


    def calculate_class_means(self, X, y):
        """
        Compute the mean vector of each class.

        Returns
        -------
        class_means : ndarray of shape (n_classes, n_features)
        """

        self.classes = np.unique(y)

        self.class_means = []

        for cls in self.classes:
            X_cls = X[y == cls]
            mean = np.mean(X_cls, axis=0)
            self.class_means.append(mean)

        self.class_means = np.array(self.class_means)

        return self.class_means


    def calculate_within_class_scatter(self, X, y):
        """
        Compute Sw

        Returns
        -------
        Sw : ndarray of shape (n_features, n_features)
        """

        self.n_features = X.shape[1]

        scatter_matrix = np.zeros((self.n_features, self.n_features))

        for cls in self.classes:

            X_cls = X[y == cls]

            class_mean = self.class_means[cls]

            diff = (X_cls - class_mean).T

            scatter_matrix += diff @ diff.T

        return scatter_matrix


    def calculate_between_class_scatter(self, X, y):
        """
        Compute Sb

        Returns
        -------
        Sb : ndarray of shape (n_features, n_features)
        """

        between_matrix = np.zeros((self.n_features,
                                        self.n_features))

        for cls in self.classes:

            class_points = X[y == cls]

            class_mean = self.class_means[cls]

            diff = (class_mean - self.overall_mean).reshape(-1, 1)

            between_matrix += (
                class_points.shape[0]
                * diff @ diff.T
            )

        return between_matrix


    def solve_eigen_problem(self,Sw,Sb):
        """
        Solve

            inv(Sw) @ Sb

        and obtain eigenvalues and eigenvectors.
        """

        matrix = np.linalg.pinv(Sw) @ Sb

        self.eigenvalues, self.eigenvectors = np.linalg.eig(matrix)

        return self.eigenvalues, self.eigenvectors


    def sort_eigenvectors(self,eigenvalues,eigenvectors):
        """
        Sort eigenvalues and eigenvectors in descending order of eigenvalues.
        """

        idx = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[idx]


        eigenvectors = eigenvectors[:, idx]

        return eigenvalues, eigenvectors


    def select_components(self,eigenvectors,n_components):
        """
        Choose the first n_components eigenvectors.
        """

        if n_components is None:
            n_components = len(self.classes) - 1

        components = eigenvectors[:, :n_components]

        return components



    def fit(self, X, y):
        """
        Learn the projection matrix.
        """
        self.calculate_overall_mean(X)
        self.calculate_class_means(X,y)
        Sw = self.calculate_within_class_scatter(X,y)
        Sb = self.calculate_between_class_scatter(X,y)
        eigenvalues,eigenvectors = self.solve_eigen_problem(Sw,Sb)
        eigenvalues,eigenvectors = self.sort_eigenvectors(eigenvalues,eigenvectors)
        self.components = self.select_components(eigenvectors,self.n_components)
        return self


    def transform(self, X):
        """
        Project data onto the learned subspace.
        """
        return  (X @ self.components)


    def fit_transform(self):
        return self.fit(X,y).transform(X)


    def discriminant_function(self, X):
        """
        Compute class discriminant scores.
        """
        pass


    def predict(self, X):
        """
        Predict class labels.
        """
        pass


    def predict_proba(self, X):
        """
        Return class probabilities/confidence scores.
        """
        pass