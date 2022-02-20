#!/usr/bin/env python3
import argparse
import os
import sys
import urllib.request

import numpy as np
import sklearn.base
import sklearn.linear_model
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing


class MNIST:
    """MNIST Dataset.
    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in the range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset, i.e., data and optionally target.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(np.float)

class PCATransformer(sklearn.base.TransformerMixin):
    def __init__(self, n_components, seed):
        self._n_components = n_components
        self._seed = seed

    def fit(self, X, y=None):
        generator = np.random.RandomState(self._seed)

        # Compute the args._n_components principal components
        # and store them as columns of self._V matrix.

        if self._n_components <= 10:
            # Use the power iteration algorithm for <= 10 dimensions.
            #
            # To compute every eigenvector, apply 10 iterations, and set
            # the initial value of every eigenvector to
            #   generator.uniform(-1, 1, size=X.shape[1])
            # Compute the vector norms using np.linalg.norm.
            n_principal_components = []
            mean = np.mean(X, axis=0)
            S = np.transpose(X - mean) @ (X - mean) / len(X)
            for _ in range(self._n_components):
                dominant_eigenvector = generator.uniform(-1, 1, size=X.shape[1])
                for iteration in range(10):
                    dominant_eigenvector = S @ dominant_eigenvector
                    dominant_eigenvalue = np.linalg.norm(dominant_eigenvector)
                    dominant_eigenvector /= dominant_eigenvalue

                n_principal_components.append(dominant_eigenvector)
                removal_matrix = dominant_eigenvalue * np.outer(dominant_eigenvector,dominant_eigenvector)
                S -= removal_matrix

            self._V = np.transpose(n_principal_components)

        else:
            # Use the SVD decomposition computed with np.linalg.svd
            U,D,V = np.linalg.svd(X-np.mean(X, axis=0))
            n_principal_components = np.transpose(V[:self._n_components])
            self._V = n_principal_components


        # We round the principal components to avoid rounding errors during
        # ReCodEx evaluation.
        self._V = np.around(self._V, decimals=4)

        return self

    def transform(self, X):
        # Transform the given X using the precomputed self._V.
        return X @ self._V

def main(args: argparse.Namespace) -> float:
    # Use the MNIST dataset.
    dataset = MNIST(data_size=args.data_size)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        dataset.data, dataset.target, test_size=args.test_size, random_state=args.seed)

    pca = [("PCA", PCATransformer(args.pca, args.seed))] if args.pca else []

    pipeline = sklearn.pipeline.Pipeline(
        [("scaling", sklearn.preprocessing.MinMaxScaler())] +
        pca +
        [("classifier", sklearn.linear_model.LogisticRegression(solver=args.solver, max_iter=args.max_iter, random_state=args.seed))]
    )
    pipeline.fit(train_data, train_target)

    test_accuracy = pipeline.score(test_data, test_target)
    return test_accuracy