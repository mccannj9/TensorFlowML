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
    from datetime import datetime

    input_layer_size = Xdata.shape[1] # should be 400
    hidden_layer_size = 50
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
    alpha = tf.constant(0.1, dtype=tf.float64, name="alpha")

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
        optimizer = tf.train.GradientDescentOptimizer(learn_rate)
        # optimizer = tf.train.AdamOptimizer(learn_rate)
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
        accuracy = tf.constant(100, dtype=tf.int64)*tf.reduce_sum(correct)/tf.shape(X, out_type=tf.int64)[0]

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    now = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
    root_logdir = "./tflow_logs"
    logdir = "%s/run-%s/" % (root_logdir, now)

    graph = tf.get_default_graph()

    cost_summary = tf.summary.scalar('COST', J)
    graph_output = tf.summary.FileWriter(logdir, graph)


    graph.finalize()

    nepochs = 1000
    batch_size = 1500

    with tf.Session() as sesh:
        init.run()
        for epoch in range(nepochs):
            nbatches = Xt.shape[0] // batch_size
            for iter in range(nbatches):
                Batch = np.random.permutation(Xt.shape[0])[:batch_size]
                Xbatch, ybatch = Xt[Batch], yt[Batch]
                # Xbatch, ybatch = Xt, yt

                if iter % 10 == 0:
                    summary_str = cost_summary.eval(feed_dict={X:Xbatch, y:ybatch})
                    step = epoch*nbatches + iter
                    graph_output.add_summary(summary_str, step)

                sesh.run(training_op, feed_dict={X:Xbatch, y:ybatch})

            cost_train = J.eval(feed_dict={X:Xt, y:yt})
            cost_test = J.eval(feed_dict={X:Xv, y:yv})
            # accuracy_tvals = accuracy.eval(feed_dict={X:Xt, y:yt})
            # accuracy_vvals = accuracy.eval(feed_dict={X:Xv, y:yv})

            print("Epoch %s ::: J=%s ::: J=%s" % (epoch, cost_train, cost_test))
            # print("Epoch %s ::: P1acc=%s ::: P2acc=%s" % (epoch, accuracy_tvals, accuracy_vvals))


            if epoch % 1000 == 0:
                save_path = saver.save(sesh, "./checkpoints/MLP_intermediate.ckpt")


        weights0 = sesh.run(W1)
        weights1 = sesh.run(W2)
        bias1 = sesh.run(b1)
        bias2 = sesh.run(b2)

        outputs = [weights0, weights1, bias1, bias2]
        prefix = [
            "DNN_hidden_kernel", "DNN_predictions_kernel",
            "DNN_hidden_bias", "DNN_predictions_bias"
        ]

        zipper = zip(outputs, prefix)

        print(logdir)
        for x,y in zipper:
            fout = "%s/%s.txt" % (logdir, y)
            np.savetxt(fname=fout, X=x, delimiter="\t")


        save_path = saver.save(sesh, "./checkpoints/MLP_final.ckpt")


def main():

    # features_path = "/media/jamc/Sticky/MachineLearning/ML_Assignments/machine-learning-ex4/ex4/Xdata.txt"
    # labels_path = "/media/jamc/Sticky/MachineLearning/ML_Assignments/machine-learning-ex4/ex4/ydata_zeros.txt"
    features_path = "/mnt/Data/GitHub/TensorFlowML/data/MNIST_training_images.txt"
    labels_path = "/mnt/Data/GitHub/TensorFlowML/data/MNIST_training_labels.txt"
    # 5000 x 400
    Xdata = np.loadtxt(features_path)
    Xdata *= 1/255
    # 5000 x 1
    ydata = np.loadtxt(labels_path)

    # does conversion of 0,1,...,9 to binary matrix 5000 x 10 after transpose
    ydata = np.array([ np.where(ydata == x, 1, 0) for x in range(0,10) ]).T

    tensorflow_perceptron(
        Xdata, ydata, diff="automatic"
    )

if __name__ == '__main__':
    main()
