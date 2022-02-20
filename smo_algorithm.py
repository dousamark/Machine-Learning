#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

# We implement the SMO algorithm as a separate method, so we can use
# it in the svm_multiclass assignment too.
def smo(
    args: argparse.Namespace,
    train_data: np.ndarray, train_target: np.ndarray,
    test_data: np.ndarray, test_target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, list[float], list[float]]:

    def kernel(args: argparse.Namespace, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if(args.kernel=="poly"):
                #(gamma * x^T y + 1) ^ degree
                return (args.kernel_gamma * (x @ y) + 1) ** args.kernel_degree
        elif(args.kernel=="rbf"):
                # exp^{- gamma * ||x - y||^2}
                return np.exp(-(args.kernel_gamma)*(np.linalg.norm(x-y)**2))
        else:
                return Exception.Yay

    def checkIfKKT(a_i,t_i,epsilon_i,args):
        return ((a_i < args.C-args.tolerance and t_i*epsilon_i< -args.tolerance) or (a_i > args.tolerance and t_i*epsilon_i>args.tolerance))

    def y(x):
        return sum(a[i]*train_target[i]*kernel_matrix[x,i] for i in range(size_train)) + b

    def predict(x):
        return sum(a[i]*train_target[i]*kernel(args,x,train_data[i]) for i in range(size_train)) + b

    # Create initial weights
    a, b = np.zeros(len(train_data)), 0
    generator = np.random.RandomState(args.seed)

    passes_without_as_changing = 0
    train_accs, test_accs = [], []

    size_train = len(train_data)
    kernel_matrix = np.zeros((size_train, size_train))
    for i in range(size_train):
        for j in range(size_train):
            kernel_matrix[i,j] = kernel(args, train_data[i], train_data[j])

    for _ in range(args.max_iterations):
        as_changed = 0
        # Iterate through the data
        for i, j in enumerate(generator.randint(len(a) - 1, size=len(a))):
            # We want j != i, so we "skip" over the value of i
            j = j + (j >= i)

            epsilon_i = y(i)-train_target[i]

            if checkIfKKT(a[i],train_target[i],epsilon_i,args):
                lagrangean = 2*kernel_matrix[i,j] - kernel_matrix[i,i] - kernel_matrix[j,j]
                if (lagrangean <= -args.tolerance):
                    epsilon_j = y(j)-train_target[j]
    
                    new_a_j = a[j] - train_target[j]*(epsilon_i - epsilon_j)/lagrangean
                
                    
                    L, H = 0,0
                    if(train_target[i] == train_target[j]):
                        L=max(0, a[i] + a[j] - args.C)
                        H = min(args.C, a[i] + a[j])
                    else:
                        L = max(0, a[j] - a[i])
                        H = min(args.C, args.C + a[j] - a[i])
                    
                    new_a_j = min(max(new_a_j,L),H)
                    new_a_i = a[i] - train_target[i]*train_target[j]*(new_a_j - a[j])

                    if abs(new_a_j - a[j]) >= args.tolerance:
                        b_j = b - epsilon_j - train_target[i]*(new_a_i - a[i])*kernel_matrix[i,j] - train_target[j]*(new_a_j - a[j])*kernel_matrix[j,j]
                        b_i = b - epsilon_i - train_target[i]*(new_a_i - a[i])*kernel_matrix[i,i] - train_target[j]*(new_a_j - a[j])*kernel_matrix[j,i]
        
                        if args.tolerance < new_a_i and new_a_i < args.C - args.tolerance:
                            b = b_i
                        elif args.tolerance < new_a_j and new_a_j < args.C - args.tolerance:
                            b = b_j
                        else:
                            b = (b_i + b_j)/2

                        a[i] = new_a_i
                        a[j] = new_a_j

                        as_changed += 1 
                    

        # After each iteration, measure the accuracy for both the
        # train set and the test set and append it to train_accs and test_accs.

        train_pred = np.sign([y(i) for i in range(len(train_data))])
        test_pred = np.sign([predict(test_dato) for test_dato in test_data])

        train_accs.append(sklearn.metrics.accuracy_score(train_target,train_pred))
        test_accs.append(sklearn.metrics.accuracy_score(test_target,test_pred))

        # Stop training if max_passes_without_as_changing passes were reached
        passes_without_as_changing = 0 if as_changed else passes_without_as_changing + 1
        if passes_without_as_changing >= args.max_passes_without_as_changing:
            break

        if len(train_accs) % 100 == 0 and len(train_accs) < args.max_iterations:
            print("Iteration {}, train acc {:.1f}%, test acc {:.1f}%".format(
                len(train_accs), 100 * train_accs[-1], 100 * test_accs[-1]))

    # Create an array of support vectors (in the same order in which they appeared
    # in the training data; to avoid rounding errors, consider a training example
    # a support vector only if a_i > args.tolerance) and their weights (a_i * t_i).
    support_vectors = []
    support_vector_weights = []
    for i in range(len(train_data)):
        if a[i] > args.tolerance:
            support_vectors.append(train_data[i])
            support_vector_weights.append(a[i]*train_target[i])

    print("Done, iteration {}, support vectors {}, train acc {:.1f}%, test acc {:.1f}%".format(
        len(train_accs), len(support_vectors), 100 * train_accs[-1], 100 * test_accs[-1]))

    return support_vectors, support_vector_weights, b, train_accs, test_accs

def main(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float, list[float], list[float]]:
    # Generate an artifical regression dataset, with +-1 as targets
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)
    target = 2 * target - 1

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Run the SMO algorithm
    support_vectors, support_vector_weights, bias, train_accs, test_accs = smo(
        args, train_data, train_target, test_data, test_target)

    if args.plot:
        import matplotlib.pyplot as plt
        def plot(predict, support_vectors):
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
            predictions = [[predict(np.array([x, y])) for x in xs] for y in ys]
            test_mismatch = np.sign([predict(x) for x in test_data]) != test_target
            plt.figure()
            plt.contourf(xs, ys, predictions, levels=0, cmap=plt.cm.RdBu)
            plt.contour(xs, ys, predictions, levels=[-1, 0, 1], colors="k", zorder=1)
            plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="o", label="Train", cmap=plt.cm.RdBu, zorder=2)
            plt.scatter(support_vectors[:, 0], support_vectors[:, 1], marker="o", s=90, label="Support Vectors", c="#00dd00")
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, marker="*", label="Test", cmap=plt.cm.RdBu, zorder=2)
            plt.scatter(test_data[test_mismatch, 0], test_data[test_mismatch, 1], marker="*", s=130, label="Test Errors", c="#ffff00")
            plt.legend(loc="upper center", ncol=4)

        predict_function = lambda x: None

        plot(predict_function, support_vectors)
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return support_vectors, support_vector_weights, bias, train_accs, test_accs