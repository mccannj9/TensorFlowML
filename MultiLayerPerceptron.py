#! /usr/bin/env python

import numpy as np

def tensorflow_perceptron(Xdata, ydata, diff="automatic"):

    import tensorflow as tf
    input_layer_size = Xdata.shape[1] # should be 400
    hidden_layer_size = 25
    num_labels = ydata.size[1] # should be 10

    # half examples for training
    Xt, yt = Xdata[::2], ydata[::2]
    # Xt, yt = Xdata, ydata
    # half examples for validation
    Xv, yv = Xdata[1::2], ydata[1::2]

    Xt = np.append(np.ones((Xt.shape[0], 1)), Xt, 1)
    Xv = np.append(np.ones((Xv.shape[0], 1)), Xv, 1)

    niters = 1000000

    X = tf.placeholder(dtype=tf.float64, name="X")
    y = tf.placeholder(dtype=tf.float64, name="y")
    # should be 401 x 25
    W1 = tf.Variable(
        tf.zeros(
            shape=(Xt.shape[1], hidden_layer_size), dtype=tf.float64
        ), name="W1"
    )

    # 2500 x 25
    Z2 = tf.matmul(X, W1, name="Z2")
    A2 = tf.divide(1, 1 + tf.exp(-Z2), name="A2")

    # should be 26 x 10
    W2 = tf.Variable(
        tf.zeros(
            shape=(hidden_layer_size, num_labels), dtype=tf.float64
        ), name="W2"
    )

    B2 = tf.Variable(
        tf.zeros(
            shape=(Xt.shape[1], hidden_layer_size), dtype=tf.float64
        ), name="B2"
    )

    Z3 = tf.matmul(A2, W2)

    learn_rate = tf.constant(0.00115, dtype=tf.float64, name="learn_rate")
    regularization = tf.constant(1, dtype=tf.float64, name="regularization")

    predictions = tf.matmul(X, weights, name="predictions")
    sigmoid = tf.divide(1, 1 + tf.exp(-1*predictions))

    res_p1 = tf.matmul(tf.transpose(y), tf.log(sigmoid))
    res_p2 = tf.matmul(tf.transpose(1-y), tf.log(1-sigmoid))

    residuals = res_p1 + res_p2

    cost = (-1/examples)*tf.reduce_sum(residuals)

    cost_gradient = tf.gradients(cost, [weights])[0]



def main():

    features_path = "/media/jamc/Sticky/MachineLearning/ML_Assignments/machine-learning-ex4/ex4/Xdata.txt"
    labels_path = "/media/jamc/Sticky/MachineLearning/ML_Assignments/machine-learning-ex4/ex4/ydata_zeros.txt"
    # 5000 x 400
    Xdata = np.loadtxt(features_path)
    # 5000 x 1
    ydata = np.loadtxt(labels_path)

    # does conversion of 0,1,...,9 to binary matrix 5000 x 10 after transpose
    ydata = np.array([ np.where(ydata == x, 1, 0) for x in range(0,10) ]).T


if __name__ == '__main__':
    main()
