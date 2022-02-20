#!/usr/bin/env python3
import argparse
import os
import sys
import urllib.request

import numpy as np
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing

class MNIST:
    """MNIST Dataset.
    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(np.float)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = x - max(x)
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def main(args: argparse.Namespace) -> float:
    # Load MNIST data, scale it to [0, 1] and split it to train and test.
    mnist = MNIST(data_size=args.train_size + args.test_size)
    mnist.data = sklearn.preprocessing.MinMaxScaler().fit_transform(mnist.data)
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        mnist.data, mnist.target, test_size=args.test_size, random_state=args.seed)

    test_predictions = []
    for dato in test_data:
        minusedMatrix = train_data - dato
        distances = np.linalg.norm(minusedMatrix, axis=1, ord=args.p)
        kNearest = sorted(range(len(distances)), key = lambda sub: distances[sub])[:args.k]   
        kNearestDistances = distances[kNearest]
        kNearestTargets = train_target[kNearest]
        output = np.zeros(10)

        if args.weights == "uniform":
            for num in kNearestTargets:
                output[num] += 1
        
        elif args.weights == "inverse":
            for i in range(len(kNearestTargets)):
                output[kNearestTargets[i]] += 1/kNearestDistances[i]

        elif args.weights == "softmax":
            softmaxed = softmax(-kNearestDistances)
            for i in range(len(kNearestTargets)):
                output[kNearestTargets[i]] += softmaxed[i]

        test_predictions.append(np.argmax(output))

    accuracy = sklearn.metrics.accuracy_score(test_target, test_predictions)

    return accuracy