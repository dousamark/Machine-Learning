#!/usr/bin/env python3
import argparse

import numpy as np
from numpy.lib.function_base import blackman
import sklearn.metrics
from sklearn.utils.extmath import squared_norm


def main(args: argparse.Namespace):

    def precomputeKernel(x ,y):
        if(args.kernel=="poly"):
            #(gamma * x^T y + 1) ^ degree
            return (args.kernel_gamma * np.transpose(x)*y + 1) ** args.kernel_degree
        elif(args.kernel=="rbf"):
            # exp^{- gamma * ||x - y||^2}
            return np.exp(-(args.kernel_gamma)*(np.linalg.norm(x-y)**2))
        else:
            return Exception.Yay
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate an artifical regression dataset
    train_data = np.linspace(-1, 1, args.data_size)
    train_target = np.sin(5 * train_data) + generator.normal(scale=0.25, size=args.data_size) + 1

    test_data = np.linspace(-1.2, 1.2, 2 * args.data_size)
    test_target = np.sin(5 * test_data) + 1

    betas = np.zeros(args.data_size)
        
    K = np.zeros((len(train_data),len(train_data)))
    for i in range(len(train_data)):
        for j in range(len(train_data)):
            #mame jenom degree maximalni velkosti, zkusit for loop od 0 do degree a dat to jako argument fci 
            #K[i][j]=precomputeKernel(np.reshape(train_data[i],(1,-1)),np.reshape(train_data[j],(1,-1)))
            K[i][j]=precomputeKernel(train_data[i],train_data[j])

    train_rmses, test_rmses = [], []
    bias = np.mean(train_target)

    for iteration in range(args.iterations):
        permutation = generator.permutation(train_data.shape[0])

        batchCounter = 0
        sum = [0] * len(train_data)

        for i in permutation:
            batchCounter += 1
            target = train_target[i]

            for j in range(len(train_data)):
                sum[i] += betas[j] * K[i,j]
            sum[i] = (sum[i] + bias -target)/args.batch_size

            if batchCounter == args.batch_size:
                batchCounter=0
                betas -= np.dot(args.learning_rate, sum)
                betas -= (args.learning_rate *args.l2*betas)
                sum = [0] * len(train_data)


        K_test = np.zeros((len(test_target),len(train_data)))
        for i in range(len(test_target)):
            for j in range(len(train_target)):
                K_test[i][j]=precomputeKernel(test_data[i],train_data[j])

        trainPred=np.matmul(betas,np.transpose(K))+bias
        testPred=np.matmul(betas,np.transpose(K_test))+bias
        train_rmses.append(sklearn.metrics.mean_squared_error(train_target,trainPred,squared= False))
        test_rmses.append(sklearn.metrics.mean_squared_error(test_target,testPred,squared= False))

        if (iteration + 1) % 10 == 0:
            print("Iteration {}, train RMSE {:.2f}, test RMSE {:.2f}".format(
                iteration + 1, train_rmses[-1], test_rmses[-1]))

    return train_rmses, test_rmses