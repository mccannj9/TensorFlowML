#! /usr/bin/env python

import numpy as np
import tensorflow as tf


def neural_layer(X, ninput, nneuron, name, activation=None):
    with tf.name_scope(name):
        # ninput = X.get_shape().as_list()[1]
        # ninput = int(X.get_shape()[1])
        # ninput = 400
        # ninput = int(tf.shape(X, out_type=tf.int64)[1])
        init_sd = np.sqrt(2/(ninput + nneuron))
        # truncated_normal prevents large weights
        init_normal = tf.truncated_normal(
            (ninput, nneuron), dtype=tf.float64, stddev=init_sd
        )
        W = tf.Variable(init_normal, dtype=tf.float64, name="kernel")
        b = tf.Variable(tf.zeros([nneuron], dtype=tf.float64), name="bias")
        Z = tf.matmul(X, W) + b

        if activation:
            return activation(Z)
        else:
            return Z


def tensorflow_perceptron(Xdata, ydata, diff="automatic"):

    # from tensorflow import tensorflow.nn.sigmoid as sigmoid

    input_layer_size = Xdata.shape[1] # should be 400
    hidden_layer_size = 25
    num_labels = ydata.shape[1] # should be 10

    # half examples for training
    Xt, yt = Xdata[::2], ydata[::2]
    # Xt, yt = Xdata, ydata
    # half examples for validation
    Xv, yv = Xdata[1::2], ydata[1::2]

    # Xt = np.append(np.ones((Xt.shape[0], 1)), Xt, 1)
    # Xv = np.append(np.ones((Xv.shape[0], 1)), Xv, 1)

    X = tf.placeholder(dtype=tf.float64, name="X")
    y = tf.placeholder(dtype=tf.float64, name="y")
    alpha = tf.constant(0.01, dtype=tf.float64, name="alpha")

    with tf.name_scope("DNN"):
        hidden_layer = neural_layer(
            X, input_layer_size, hidden_layer_size, "hidden", activation=tf.sigmoid
        )
        outputs = neural_layer(
            hidden_layer, hidden_layer_size, num_labels, "predictions", activation=tf.sigmoid
        )

    with tf.name_scope("Cost"):
        part1_cost = tf.multiply(y, tf.log(outputs))
        part2_cost = tf.multiply(1 - y, tf.log(1-outputs))
        total_cost = tf.add(part1_cost, part2_cost)
        mean_cost = -1*tf.reduce_mean(total_cost)
        dgraph = tf.get_default_graph()
        W1 = dgraph.get_tensor_by_name("DNN/hidden/kernel:0")
        W2 = dgraph.get_tensor_by_name("DNN/predictions/kernel:0")
        part1_reg = tf.reduce_mean(tf.square(W1))
        part2_reg = tf.reduce_mean(tf.square(W2))
        reg = (alpha) * (part1_reg + part2_reg)
        J = mean_cost + reg

    learn_rate = 0.1
    with tf.name_scope("Training"):
        # optimizer = tf.train.GradientDescentOptimizer(learn_rate)
        optimizer = tf.train.AdamOptimizer(learn_rate)
        training_op = optimizer.minimize(J)

    with tf.name_scope("FeedForward"):
        b1 = dgraph.get_tensor_by_name("DNN/hidden/bias:0")
        b2 = dgraph.get_tensor_by_name("DNN/predictions/bias:0")
        z2 = tf.matmul(X, W1) + b1
        a2 = tf.sigmoid(z2)
        z3 = tf.matmul(a2, W2) + b2
        preds = tf.argmax(z3, axis=1)
        labels = tf.argmax(y, axis=1)
        correct = tf.cast(tf.equal(preds, labels), tf.int64)
        accuracy = tf.reduce_sum(correct)/tf.shape(X, out_type=tf.int64)[0]

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    nepochs = 4000
    batch_size = 50

    with tf.Session() as sesh:
        init.run()
        for epoch in range(nepochs):
            for iter in range(Xt.shape[0] // batch_size):
                Batch = np.random.permutation(Xt.shape[0])[:batch_size]
                Xbatch, ybatch = Xt[Batch], yt[Batch]
                # Xbatch, ybatch = Xt, yt
                sesh.run(training_op, feed_dict={X:Xbatch, y:ybatch})
            cost_train = J.eval(feed_dict={X:Xt, y:yt})
            cost_test = J.eval(feed_dict={X:Xv, y:yv})
            # acc_train = accuracy.eval(feed_dict={X:Xt, y:yt})
            # acc_test = accuracy.eval(feed_dict={X:Xv, y:yv})
            # correct_tvals = correct.eval(feed_dict={X:Xt, y:yt})
            # correct_vvals = correct.eval(feed_dict={X:Xv, y:yv})
            accuracy_tvals = accuracy.eval(feed_dict={X:Xt, y:yt})
            accuracy_vvals = accuracy.eval(feed_dict={X:Xv, y:yv})

            # print("Epoch %s ::: Tacc=%s ::: Vacc=%s" % (epoch, acc_train, acc_test))
            print("Epoch %s ::: J=%s ::: J=%s" % (epoch, cost_train, cost_test))
            # part1_cost_val = part1_cost.eval(feed_dict={X:Xt, y:yt})
            # part2_cost_val = part2_cost.eval(feed_dict={X:Xt, y:yt})
            # print("Epoch %s ::: P1acc=%s ::: P2acc=%s" % (epoch, correct_tvals, correct_vvals))
            print("Epoch %s ::: P1acc=%s ::: P2acc=%s" % (epoch, accuracy_tvals, accuracy_vvals))

        save_path = saver.save(sesh, "./my_model_final.ckpt")


def main():

    features_path = "/media/jamc/Sticky/MachineLearning/ML_Assignments/machine-learning-ex4/ex4/Xdata.txt"
    labels_path = "/media/jamc/Sticky/MachineLearning/ML_Assignments/machine-learning-ex4/ex4/ydata_zeros.txt"
    # 5000 x 400
    Xdata = np.loadtxt(features_path)
    # 5000 x 1
    ydata = np.loadtxt(labels_path)

    # does conversion of 0,1,...,9 to binary matrix 5000 x 10 after transpose
    ydata = np.array([ np.where(ydata == x, 1, 0) for x in range(0,10) ]).T

    tensorflow_perceptron(
        Xdata, ydata, diff="automatic"
    )

if __name__ == '__main__':
    main()
