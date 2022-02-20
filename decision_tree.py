#!/usr/bin/env python3
import argparse
import heapq

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

class DecisionTree:
    class Node:
        def __init__(self, instances, prediction):
            self.is_leaf = True
            self.instances = instances
            self.prediction = prediction

        def split(self, feature, value, left, right):
            self.is_leaf = False
            self.feature = feature
            self.value = value
            self.left = left
            self.right = right

    def __init__(self, criterion, max_depth, min_to_split, max_leaves):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_to_split = min_to_split
        self.max_leaves = max_leaves

    def fit(self, data, targets):
        self.data = data
        self.targets = targets

        self._root = self.get_Leaf(np.arange(len(self.data)))
        if self.max_leaves is None:
            self.recursive_splitting(self._root, 0)
        else:
            self.adaptive_splitting()

    def predict(self, data):
        results = np.zeros(len(data), dtype=np.int32)
        for i in range(len(data)):
            node = self._root
            while not node.is_leaf:
                node = node.left if data[i][node.feature] <= node.value else node.right
            results[i] = node.prediction

        return results

    def recursive_splitting(self, node, depth):
        if not self.check_splitting(node, depth):
            return

        _, feature, value, left, right = self.find_best_split(node)
        node.split(feature, value, self.get_Leaf(left), self.get_Leaf(right))
        self.recursive_splitting(node.left, depth + 1)
        self.recursive_splitting(node.right, depth + 1)

    def adaptive_splitting(self):
        def split_value(node, index, depth):
            best_split = self.find_best_split(node)
            return (best_split[0], index, depth, node, *best_split[1:])

        heap = [split_value(self._root, 0, 0)]
        for i in range(self.max_leaves - 1):
            _, _, depth, node, feature, value, left, right = heapq.heappop(heap)
            node.split(feature, value, self.get_Leaf(left), self.get_Leaf(right))
            if self.check_splitting(node.left, depth + 1):
                heapq.heappush(heap, split_value(node.left, 2 * i + 1, depth + 1))
            if self.check_splitting(node.right, depth + 1):
                heapq.heappush(heap, split_value(node.right, 2 * i + 2, depth + 1))
            if not heap:
                break

    def check_splitting(self, node, depth):
        #depth is sufficient and size is sufficient
        return ((self.max_depth is None or depth < self.max_depth) and len(node.instances) >= self.min_to_split)

    def find_best_split(self, node):
        best_criterion = None
        for feature in range(self.data.shape[1]):
            node_features = self.data[node.instances, feature]
            separators = np.unique(node_features)
            for i in range(len(separators) - 1):
                value = (separators[i] + separators[i + 1]) / 2
                left, right = node.instances[node_features <= value], node.instances[node_features > value]

                if(self.criterion=="gini"):
                    criterion = self.get_Gini(left) + self.get_Gini(right)
                else:
                    criterion = self.get_Entropy(left) + self.get_Entropy(right)

                if best_criterion is None or criterion < best_criterion:
                    best_criterion, best_feature, best_value, best_left, best_right = \
                        criterion, feature, value, left, right
        
        crit = None
        if(self.criterion=="gini"):
            crit = self.get_Gini(node.instances)
        else:
            crit = self.get_Entropy(node.instances)

        return best_criterion - crit, best_feature, best_value, best_left, best_right

    def get_Leaf(self, instances):
        return self.Node(instances, np.argmax(np.bincount(self.targets[instances])))

    def get_Gini(self, instances):
        bins = np.bincount(self.targets[instances])
        return np.sum(bins * (1 - bins / len(instances)))

    def get_Entropy(self, instances):
        bins = np.bincount(self.targets[instances])
        bins = bins[np.nonzero(bins)]
        return 0-np.sum(bins * np.log(bins / len(instances)))

def main(args: argparse.Namespace):
    # Use the given dataset
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    decision_tree = DecisionTree(args.criterion, args.max_depth, args.min_to_split, args.max_leaves)
    decision_tree.fit(train_data, train_target)

    train_accuracy = sklearn.metrics.accuracy_score(train_target, decision_tree.predict(train_data))
    test_accuracy = sklearn.metrics.accuracy_score(test_target, decision_tree.predict(test_data))
    return train_accuracy, test_accuracy