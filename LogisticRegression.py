#! /usr/bin/env python

import numpy as np


def tensorflow_solution_gradient_descent(Xdata, ydata, diff="automatic"):

    import tensorflow as tf

    # half examples for training
    Xt, yt = Xdata[::2], ydata[::2]
    # Xt, yt = Xdata, ydata
    # half examples for validation
    Xv, yv = Xdata[1::2], ydata[1::2]

    Xt = np.append(np.ones((Xt.shape[0], 1)), Xt, 1)
    Xv = np.append(np.ones((Xv.shape[0], 1)), Xv, 1)
    # print(Xt)

    niters = 1000000

    X = tf.placeholder(dtype=tf.float64, name="X")
    y = tf.placeholder(dtype=tf.float64, name="y")
    weights = tf.Variable(tf.zeros(shape=(Xt.shape[1],1), dtype=tf.float64), name="weights")

    learn_rate = tf.constant(0.00115, dtype=tf.float64, name="learn_rate")
    examples = tf.constant(Xt.shape[0], dtype=tf.float64, name="examples")

    predictions = tf.matmul(X, weights, name="predictions")
    sigmoid = tf.divide(1, 1 + tf.exp(-1*predictions))

    res_p1 = tf.matmul(tf.transpose(y), tf.log(sigmoid))
    res_p2 = tf.matmul(tf.transpose(1-y), tf.log(1-sigmoid))

    residuals = res_p1 + res_p2

    cost = (-1/examples)*tf.reduce_sum(residuals)
    # cost = tf.reduce_mean(tf.log(1+tf.exp(-y*predictions)))

    if diff == "automatic":
        print("Using automatic differentiation for gradient descent")
        cost_gradient = tf.gradients(cost, [weights])[0] # automatic differentiation
    else:
        print("Using closed-form gradient for gradient descent")
        XT = tf.transpose(X, name="XT")
        cost_gradient = 1/Xt.shape[0] * tf.matmul(XT, sigmoid-y)

    update_weights = weights.assign(weights - learn_rate * cost_gradient)

    init = tf.global_variables_initializer()

    config = tf.ConfigProto(
        device_count = {'CPU': 1}
    )

    saver = tf.train.Saver()

    graph = tf.get_default_graph()
    graph.finalize()

    feeder = {X: Xt, y: yt}


    with tf.Session(config=config) as sesh:
        sesh.run(init)
        saver.save(sesh, './logreg_test', global_step=10000)

        for i in range(niters):
            weights_value = sesh.run(update_weights, feed_dict=feeder)
            if i % 1000 == 0:
                cgrad = sesh.run(cost_gradient, feed_dict=feeder)
                train_cost = sesh.run(cost, feed_dict=feeder)
                valid_cost = sesh.run(cost, feed_dict=feeder)
                print(
                    "Iteration %s :: Train Cost %s :: Valid Cost %s" % (
                        i, train_cost, valid_cost
                    )
                )
                # print(cgrad)

        print("First few weights = ", weights_value[:5].T)
        print("Cost on training data = ", train_cost)
        print("Cost on validation data = ", valid_cost)

    return weights

def main():
    import os, sys
    data_dir = os.path.dirname(os.path.abspath(__file__)) + "/data/"

    # features_path = data_dir + "Xtrain.txt"
    features_path = "/media/jamc/Sticky/MachineLearning/DeepLearning/data/Xtrain.txt"
    labels_path = "/media/jamc/Sticky/MachineLearning/DeepLearning/data/Ytrain_zeros.txt"
    Xdata = np.loadtxt(features_path)
    ydata = np.loadtxt(labels_path)
    ydata = ydata.reshape(-1,1)

    tf_gradient_wvector = tensorflow_solution_gradient_descent(
        Xdata, ydata, diff="automatic"
    )
    #
    # tf_gradient_wvector = tensorflow_solution_gradient_descent(
    #     Xdata, ydata, diff="closed-form"
    # )

    # # Some additional test data from Coursera course on LogReg
    # features_path = "/media/jamc/Sticky/ML_Assignments/machine-learning-ex2/ex2/ex2data1_X.txt"
    # # features_path = "/media/jamc/Sticky/ML_Assignments/machine-learning-ex1/week3_functions/data_X.txt"
    # labels_path = "/media/jamc/Sticky/ML_Assignments/machine-learning-ex2/ex2/ex2data1_y.txt"
    # # labels_path = "/media/jamc/Sticky/ML_Assignments/machine-learning-ex1/week3_functions/data_y_negs.txt"
    # #
    # Xdata = np.loadtxt(features_path)
    # ydata = np.loadtxt(labels_path)
    # ydata = ydata.reshape(-1,1)
    #
    # tf_gradient_wvector = tensorflow_solution_gradient_descent(
    #     Xdata, ydata, diff="automatic"
    # )

    # tf_gradient_wvector = tensorflow_solution_gradient_descent(
    #     Xdata, ydata, diff="closed-form"
    # )

if __name__ == '__main__':
    main()
