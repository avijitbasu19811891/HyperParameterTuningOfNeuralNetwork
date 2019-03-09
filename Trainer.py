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
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

from Params import WeightFileName
import keras

from keras import layers, initializers

from Params import ResultFileName

from Params import isKerasVerbose

from Params import EnableKerasDebug
from Params import reTrainExistingNetworks

import sys

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
fResult = open(ResultFileName, "w")


from DebugUtils import GlobalTrainingTrend

from DebugUtils import KerasPlotModel

def trainKeras(nn, train_data, train_labels, test_data=None, test_labels=None):
    keras.initializers.lecun_uniform(seed=None)
    print(train_data.shape[1])
    # Get our network parameters.
    nb_layers = nn._network['nb_layers']
    nb_neurons = nn._network['nb_neurons']
    activation = nn._network['activation']
    optimizer = nn._network['optimizer']

    """
    if reTrainExistingNetworks == False:
        if nn.isTrained() == True:
    """

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation='relu', input_shape=(train_data.shape[1],),
                            kernel_initializer='random_uniform',
                            bias_initializer=initializers.Constant(0.1)))
        else:
            model.add(Dense(nb_neurons, activation=activation,
                            kernel_initializer='random_uniform',
                            bias_initializer=initializers.Constant(0.1)))

    """
       This model is already trained. Start from the old weight
    """
    if reTrainExistingNetworks == False:
        if nn.isWeightSet() != 0:
            print("Reusing old weights:")
            model.set_weights(nn.getWeight())


    # Output layer.
    model.add(Dense(1, activation=None))

    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])



    labels = []
    for label in train_labels:
        l = np.zeros(num_classes)
        l[int(label)] = 1.0
        labels.append(l)
    labels = np.array(labels)

    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.weights = []
            self.n = 0
            self.n += 1

        def on_epoch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            w = model.get_weights()
            self.weights.append([x.flatten()[0] for x in w])
            self.n += 1

    callbackList = []
    callbackList.append(ModelCheckpoint(WeightFileName, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True,
                        mode='auto', period=1))
    callbackList.append(EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto', baseline=None, restore_best_weights=False))



    if EnableKerasDebug == True:
        history = LossHistory()
        callbackList.append(history)
        callbackList.append(TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch'))

    print("Initiating training")
    modelHistory = model.fit(train_data, train_labels,
              batch_size=10,
              epochs=150,  # using early stopping, so no real limit
              verbose=isKerasVerbose,
              validation_data=(test_data, test_labels),
              callbacks= callbackList)

    weight = model.get_weights()
    #print(weight)
    print("End of  training")

    """
       Try to evaluate this trained model, on the test data.
       this will evaluate the accuracy of this model.
    """
    score = model.evaluate(test_data, test_labels, verbose=isKerasVerbose)

    print(score)

    nn.updateTrainingHistory(score[1], score[0])

    """
       Update the weight only when the accuracy has improved.
    """

    if (nn.accuracy() < score[1]):
        json_string = model.to_json()
        print(json_string)
        GlobalTrainingTrend.updateIncInAccuracy(nn.accuracy(), score[1])
        nn.updateWeight(weight)
        nn.updateAccuracy(score[1])
        nn.updateSummary(model.summary())
    else:
        GlobalTrainingTrend.updateDecInAccuracy(nn.accuracy(), score[1])

    if EnableKerasDebug == True:
        print(history.losses)
        model.summary()
        #loss = history.losses
        #GlobalTrainingTrend.update(loss)


    KerasPlotModel(modelHistory, nn.nnName())

    return score[1]  # 1 is accuracy. 0 is loss.



