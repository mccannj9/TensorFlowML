#! /usr/bin/env python

import numpy as np


def numpy_solution(Xdata, ydata):

    # half examples for training
    Xtrain, ytrain = Xdata[::2], ydata[::2]
    # half examples for validation
    Xvalid, yvalid = Xdata[1::2], ydata[1::2]

    # closed-form solution to minimize cost (loss) function
    # min(cost) = inverse(XT*X) * (XT*y)
    XT_X = np.dot(Xtrain.T, Xtrain)
    XT_y = np.dot(Xtrain.T, ytrain)
    weights = np.dot(np.linalg.inv(XT_X), XT_y)

    print("First few weights = ", weights[:5].T)

    train_predictions = np.dot(Xtrain, weights)
    train_cost = np.mean(np.square(train_predictions-ytrain))
    print("Cost on training data = ", train_cost)

    valid_predictions = np.dot(Xvalid, weights)
    valid_cost = np.mean(np.square(valid_predictions-yvalid))
    print("Cost on validation data = ", valid_cost)

    return weights

def tensorflow_solution_closed_form(Xdata, ydata):

    import tensorflow as tf

    # half examples for training
    Xt, yt = Xdata[::2], ydata[::2]
    # half examples for validation
    Xv, yv = Xdata[1::2], ydata[1::2]

    X = tf.placeholder(dtype=tf.float64, name="X")
    y = tf.placeholder(dtype=tf.float64, name="y")
    weights = tf.Variable(tf.zeros(shape=(Xdata.shape[1],1), dtype=tf.float64), name="weights")

    XT = tf.transpose(X, name="XT")
    XT_X = tf.matmul(XT, X, name="XT_X")
    XT_y = tf.matmul(XT, y, name="XT_y")

    compute_weights = weights.assign(
        tf.matmul(tf.matrix_inverse(XT_X), XT_y, name="weights")
    )

    predictions = tf.matmul(X, weights, name="predictions")
    cost = tf.reduce_mean(tf.square(predictions-y))

    init = tf.global_variables_initializer()

    with tf.Session() as sesh:
        weights_value = sesh.run(compute_weights, feed_dict={X: Xt, y: yt})
        print("First few weights = ", weights_value[:5].T)

        train_cost = sesh.run(cost, feed_dict={X: Xt, y: yt})
        print("Cost on training data = ", train_cost)

        valid_cost = sesh.run(cost, feed_dict={X: Xv, y: yv})
        print("Cost on validation data = ", valid_cost)

    return weights

def tensorflow_solution_gradient_descent(Xdata, ydata, diff="automatic"):

    import tensorflow as tf

    # half examples for training
    Xt, yt = Xdata[::2], ydata[::2]
    # half examples for validation
    Xv, yv = Xdata[1::2], ydata[1::2]

    # setup for gradient descent
    niters = 10000000

    X = tf.placeholder(dtype=tf.float64, name="X")
    y = tf.placeholder(dtype=tf.float64, name="y")
    weights = tf.Variable(tf.zeros(shape=(Xdata.shape[1],1), dtype=tf.float64), name="weights")

    # penalty = tf.constant(0.001, dtype=tf.float64, name="penalty")
    learn_rate = tf.constant(0.0055, dtype=tf.float64, name="learn_rate")
    examples = tf.constant(Xdata.shape[0], dtype=tf.float64, name="examples")

    predictions = tf.matmul(X, weights, name="predictions")
    residuals = predictions-y
    cost = tf.reduce_mean(tf.square(residuals))
    # Tried some regularized linear regression here (need more info)
    # cost = tf.add(
    #     tf.reduce_mean(tf.square(residuals)),
    #     tf.multiply(tf.reduce_sum(tf.square(weights)), penalty)
    # )



    if diff == "automatic":
        print("Using automatic differentiation for gradient descent")
        cost_gradient = tf.gradients(cost, [weights])[0] # automatic differentiation
    else:
        print("Using closed-form gradient for gradient descent")
        XT = tf.transpose(X, name="XT")
        cost_gradient = 2./Xt.shape[0] * tf.matmul(XT, residuals)

    update_weights = weights.assign(weights - learn_rate * cost_gradient)
    # update_weights = weights.assign(weights*(1-penalty*learn_rate/examples) - (learn_rate/examples) * cost_gradient)


    init = tf.global_variables_initializer()
    graph = tf.get_default_graph()
    graph.finalize()

    with tf.Session() as sesh:
        sesh.run(init)

        for i in range(niters):
            weights_value = sesh.run(update_weights, feed_dict={X: Xt, y: yt})
            if i % 1000 == 0:
                train_cost = sesh.run(cost, feed_dict={X: Xt, y: yt})
                valid_cost = sesh.run(cost, feed_dict={X: Xv, y: yv})
                print(
                    "Iteration %s :: Train Cost %s :: Valid Cost %s" % (
                        i, train_cost, valid_cost
                    )
                )

        print("First few weights = ", weights_value[:5].T)
        print("Cost on training data = ", train_cost)
        print("Cost on validation data = ", valid_cost)

    return weights

def main():
    import os, sys
    data_dir = os.path.dirname(os.path.abspath(__file__)) + "/data/"

    features_path = data_dir + "Xtrain.txt"
    labels_path = data_dir + "Ytrain.txt"
    Xdata = np.loadtxt(features_path)
    ydata = np.loadtxt(labels_path)
    ydata = ydata.reshape(-1,1)

    # results from numpy (weight vector)
    np_linreg_wvector = numpy_solution(Xdata, ydata)

    tf_linreg_wvector = tensorflow_solution_closed_form(
        Xdata, ydata
    )

    tf_gradient_wvector = tensorflow_solution_gradient_descent(
        Xdata, ydata, diff="automatic"
    )




if __name__ == '__main__':
    main()
