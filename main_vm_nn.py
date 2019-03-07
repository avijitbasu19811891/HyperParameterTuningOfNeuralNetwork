from __future__ import print_function
import tensorflow as tf
import numpy as np
import glob
import csv
import matplotlib.pyplot as plt
import itertools

num_classes = 10
sample_history = 10


def load_data():
    path = "./ml_data/*.csv"
    data = []
    labels = []

    print("loading data...")

    for fname in glob.glob(path):
        with open(fname, 'r') as infh:
            reader = csv.reader(infh, delimiter=';')

            for row in reader:
                r = np.array(row, dtype=float)
                rr = []
                for i in range(sample_history):
                    rr.append(r[i * 7 + 1])
                # print(rr)
                data.append(rr)
                labels.append(r[-1])

    data = np.array(data)
    labels = np.array(labels)
    n = int(float(data.shape[0]) * 0.8)
    train_data = data[:n]
    train_labels = labels[:n]
    test_data = data[n:]
    test_labels = labels[n:]
    print("finished loading data")
    return train_data, train_labels, test_data, test_labels


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def plot_confusion_matrix(con_mat, nor=False):
    if nor:
        con_mat = con_mat.astype(float) / con_mat.sum(axis=0)
    plt.imshow(con_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(np.arange(num_classes), np.arange(num_classes))
    plt.yticks(np.arange(num_classes), np.arange(num_classes))
    fmt = '.2f' if nor else '.0f'
    threshhold = con_mat.max() / 2.
    for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
        plt.text(j, i, format(con_mat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if con_mat[i, j] > threshhold else "black")
    plt.tight_layout()
    plt.xlabel('True label')
    plt.ylabel('Predicted label')


def arnn(train_data, train_labels, test_data, test_labels):
    xs = tf.placeholder(tf.float32, [None, train_data.shape[1]])
    ys = tf.placeholder(tf.float32, [None, num_classes])
    l1 = add_layer(xs, train_data.shape[1], 100, activation_function=tf.nn.relu)
    pre_input = tf.concat([l1, xs], 1)
    out = add_layer(pre_input, 100 + train_data.shape[1], num_classes, activation_function=None)
    prediction = tf.argmax(out, axis=1)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=ys))

    train_step = tf.train.AdamOptimizer().minimize(loss)

    if int((tf.__version__).split('.')[1]) < 12:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    labels = []
    for label in train_labels:
        l = np.zeros(num_classes)
        l[int(label)] = 1.0
        labels.append(l)
    labels = np.array(labels)

    for i in range(10000):

        sess.run(train_step, feed_dict={xs: train_data, ys: labels})
        if i % 50 == 0:
            train_accuracy = np.mean(np.argmax(labels, axis=1) ==
                                     sess.run(prediction, feed_dict={xs: train_data, ys: labels}))
            print("Train Accuracy = ", train_accuracy)

    te_labels = []
    for label in test_labels:
        l = np.zeros(num_classes)
        l[int(label)] = 1.0
        te_labels.append(l)
    te_labels = np.array(te_labels)

    train_accuracy = np.mean(np.argmax(labels, axis=1) ==
                             sess.run(prediction, feed_dict={xs: train_data, ys: labels}))
    test_accuracy = np.mean(np.argmax(te_labels, axis=1) ==
                            sess.run(prediction, feed_dict={xs: test_data, ys: te_labels}))
    print("---------------")
    print("Train Accuracy = ", train_accuracy)
    print("Test Accuracy = ", test_accuracy)
    test_pred = sess.run(prediction, feed_dict={xs: test_data, ys: te_labels})

    con_mat = np.zeros((num_classes, num_classes))
    for t, l in zip(test_pred, test_labels):
        con_mat[int(t)][int(l)] += 1

    print("Confusion Matrix = {}".format(con_mat))
    plt.figure()
    plot_confusion_matrix(con_mat, nor=True)
    plt.show()


if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = load_data()
    arnn(train_data, train_labels, test_data, test_labels)