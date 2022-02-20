# b6523403-e8f0-11e9-9ce9-00505601122b
# a194809e-e8e6-11e9-9ce9-00505601122b
# 946dc1a7-eb43-11e9-9ce9-00505601122b

#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import sys
import urllib.request
import re

import numpy as np
from numpy.core.fromnumeric import size
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors


class NewsGroups:
    def __init__(
        self,
        name="20newsgroups.train.pickle",
        data_size=None,
        url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/",
    ):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        with lzma.open(name, "rb") as dataset_file:
            dataset = pickle.load(dataset_file)

        self.DESCR = dataset.DESCR
        self.data = dataset.data[:data_size]
        self.target = dataset.target[:data_size]
        self.target_names = dataset.target_names

def create_features(documents):
    features = {}
    for document in documents:
        words = re.findall("\w{2,}", document)
        for word in words:
            features[word] = features.get(word, 0) + 1
    filtered_features = {}
    for key in features:
        if features[key] > 1:
            filtered_features[key] = features[key]
    return filtered_features


def binary(documents, features):
    weights = np.ndarray((len(documents), len(features)))
    for i in range(len(documents)):
        featureDict = {}
        words = re.findall("\w{2,}", documents[i])
        for word in words:
            featureDict[word] = 1
        weights[i, :] = [featureDict.get(feature, 0) for feature in features]
    return weights


def tf(documents, features):
    tf_matrix = np.ndarray((len(documents), len(features)))
    for i in range(len(documents)):
        featureDict = {}
        words = re.findall("\w{2,}", documents[i])
        all_terms = len(words)
        for word in words:
            featureDict[word] = featureDict.get(word, 0) + 1
        tf_matrix[i, :] = [
            featureDict.get(feature, 0) / all_terms for feature in features
        ]
    return tf_matrix


def idf(documents, features, tfOrBinary):
    counts = np.sign(tfOrBinary)
    lenDocs = len(documents)
    idfMatrix = np.array(
        [(np.log10(lenDocs / (np.sum(counts[:, i]) + 1))) for i in range(len(features))]
    )
    return idfMatrix


def create_weights(tf_bool, idf_bool, data, features, idf_weights):
    tf_weights = None
    if tf_bool:
        tf_weights = tf(data, features)
    else:
        tf_weights = binary(data, features)
    if idf_bool:
        if idf_weights is None:
            idf_weights = idf(data, features, tf_weights)
        weights = np.zeros((tf_weights.shape))
        weights = tf_weights * idf_weights.reshape([1, -1])
    else:
        weights = tf_weights

    weights = np.array(
        sklearn.preprocessing.Normalizer(norm="l2").fit_transform(weights)
    )
    return weights, idf_weights


def main(args: argparse.Namespace) -> float:
    # Load the 20newsgroups data.
    newsgroups = NewsGroups(data_size=args.train_size + args.test_size)

    # Create train-test split.
    (
        train_data,
        test_data,
        train_target,
        test_target,
    ) = sklearn.model_selection.train_test_split(
        newsgroups.data,
        newsgroups.target,
        test_size=args.test_size,
        random_state=args.seed,
    )

    # Create a feature for every word that is present at least twice
    # in the training data. A word is every maximal sequence of at least 2 word characters,
    # where a word character corresponds to a regular expression `\w`.
    features = create_features(train_data)
    featuresList = list(features.keys())

    weights, idf_weights = create_weights(
        args.tf, args.idf, train_data, featuresList, None
    )

    knn = sklearn.neighbors.KNeighborsClassifier(
        n_neighbors=args.k, algorithm="brute", metric="minkowski", p=2
    )
    knn.fit(weights, train_target)

    test_features, _ = create_weights(
        args.tf, args.idf, test_data, featuresList, idf_weights
    )

    predictions = knn.predict(test_features)
    f1_score = sklearn.metrics.f1_score(test_target, predictions, average="macro")

    return f1_score