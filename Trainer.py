import tensorflow as tf
import numpy as np
from NNPorpulation import NeuralNetwork

num_classes = 10

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

def arnn(nn, train_data, train_labels):
    xs = tf.placeholder(tf.float32, [None, train_data.shape[1]])
    ys = tf.placeholder(tf.float32, [None, num_classes])
    activation_function = None
    l1 = None
    l2 = None
    l3 = None
    out = None
    pre_input = None
    loss = None

    if nn._network['activation'] == 'tanh':
        activation_function = tf.nn.tanh
    elif nn._network['activation'] == 'relu':
        activation_function = tf.nn.relu
    elif nn._network['activation'] == 'elu':
        activation_function = tf.nn.elu
    if nn._network['nb_layers'] == 1:
        l1 = add_layer(xs, train_data.shape[1], nn._network['nb_neurons'], activation_function)
        pre_input = tf.concat([l1, xs], 1)
        out = add_layer(pre_input,  nn._network['nb_neurons'] + train_data.shape[1], num_classes, activation_function = None)
    if nn._network['nb_layers'] == 2:
        l1 = add_layer(xs, train_data.shape[1], nn._network['nb_neurons'], activation_function)
        input = tf.concat([l1,xs],1)
        l2 = add_layer(input, nn._network['nb_neurons'] + train_data.shape[1], train_data.shape[1],  activation_function)
        pre_input = tf.concat([l2, xs], 1)
        out = add_layer(pre_input, nn._network['nb_neurons'] + train_data.shape[1], num_classes, activation_function=None)
    elif nn._network['nb_layers'] == 3:
        l1 = add_layer(xs, train_data.shape[1], nn._network['nb_neurons'], activation_function)
        l2 = add_layer(tf.concat([l1,xs],1),nn._network['nb_neurons'] + train_data.shape[1], num_classes, activation_function)
        l3 = add_layer(tf.concat([l2,xs],1), nn._network['nb_neurons'] + train_data.shape[1], num_classes, activation_function)
        pre_input = tf.concat([l3, xs], 1)
        out = add_layer(pre_input, nn._network['nb_neurons'] + train_data.shape[1], num_classes, activation_function=None)
    elif nn._network['nb_layers'] == 4:
        l1 = add_layer(xs, train_data.shape[1], nn._network['nb_neurons'], activation_function)
        l2 = add_layer(tf.concat([l1,xs],1), nn._network['nb_neurons'] + train_data.shape[1], num_classes, activation_function)
        l3 = add_layer(tf.concat([l2,xs],1), nn._network['nb_neurons'] + train_data.shape[1], num_classes, activation_function)
        l4 = add_layer(tf.concat([l3,xs],1), nn._network['nb_neurons'] + train_data.shape[1], num_classes, activation_function)
        pre_input = tf.concat([l4, xs], 1)
        out = add_layer(pre_input, nn._network['nb_neurons'] + train_data.shape[1], num_classes, activation_function=None)
    else:
        print("error")
    prediction = tf.argmax(out, axis=1)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=ys))

    train_step = tf.train.AdamOptimizer().minimize(loss)
    if nn._network['optimizer'] == 'sgd':
        train_step = tf.keras.optimizers.SGD.minimize(loss)
    elif nn._network['optimizer'] == 'rmsprop':
        train_step = tf.train.RMSPropOptimizer().minimize(loss)
    elif nn._network['optimizer'] == 'adagrad':
        train_step = tf.train.AdagradDAOptimizer().minimize(loss)

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

    return train_accuracy


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
import keras

def trainKeras(nn, train_data, train_labels, test_data=None, test_labels=None):
    keras.initializers.lecun_uniform(seed=None)
    print(train_data.shape[1])
    # Get our network parameters.
    nb_layers = nn._network['nb_layers']
    nb_neurons = nn._network['nb_neurons']
    activation = nn._network['activation']
    optimizer = nn._network['optimizer']

    model = Sequential()
    print(train_data.shape[1])
    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation='sigmoid', input_shape=(train_data.shape[1],)))
        else:
            model.add(Dense(nb_neurons, activation=activation))



    # Output layer.
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])



    labels = []
    for label in train_labels:
        l = np.zeros(num_classes)
        l[int(label)] = 1.0
        labels.append(l)
    labels = np.array(labels)

    print("Initiating training")
    model.fit(train_data, train_labels,
              batch_size=10,
              epochs=150,  # using early stopping, so no real limit
              verbose=1,
              validation_data=(test_data, test_labels),
              callbacks=[EarlyStopping(patience=5)])

    print("End of  training")
    score = model.evaluate(test_data, test_labels, verbose=1)

    print(score)
    return score[1]  # 1 is accuracy. 0 is loss.



