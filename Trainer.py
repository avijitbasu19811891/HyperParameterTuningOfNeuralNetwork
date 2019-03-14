import tensorflow as tf
import numpy as np
from NNPorpulation import NeuralNetwork

num_classes = 10

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
from Params import KerasEnableDropOut

import sys

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
fResult = open(ResultFileName, "w")


from DebugUtils import GlobalTrainingTrend

from DebugUtils import KerasPlotModel

from DebugUtils import PrintConfusionMatrix

from keras.metrics import categorical_accuracy

def trainKeras(nn, train_data, train_labels, test_data=None, test_labels=None):

    xTest_bak = test_data
    yTest_bak = test_labels

    yTrainCat = keras.utils.to_categorical(train_labels, num_classes=10, dtype='float32')

    yTestCat = keras.utils.to_categorical(test_labels, num_classes=10, dtype='float32')
    print("Shape:")
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
            model.add(Dense(nb_neurons, activation=None, input_shape=(train_data.shape[1],),
                            kernel_initializer='random_uniform',
                            bias_initializer=initializers.Constant(0.1)))
        else:
            model.add(Dense(nb_neurons, activation=activation,
                            kernel_initializer='random_uniform',
                            bias_initializer=initializers.Constant(0.1)))
            if KerasEnableDropOut == True:
                if activation == 'relu':
                    model.add(keras.layers.AlphaDropout(0.2))
                if activation == 'selu':
                    model.add(keras.layers.AlphaDropout(0.2))
    # Output layer.
    model.add(Dense(10, activation='softmax'))

    """
       This model is already trained. Start from the old weight
    """
    if reTrainExistingNetworks == False:
        if nn.isWeightSet() != 0:
            print("Reusing old weights:")
            model.set_weights(nn.getWeight())

    tf.keras.backend.set_learning_phase(1)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

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
    modelHistory = model.fit(train_data, yTrainCat,
              batch_size=10,
              epochs=150,  # using early stopping, so no real limit
              verbose=isKerasVerbose,
              validation_data=(test_data, yTestCat),
              callbacks= callbackList)

    weight = model.get_weights()
    #print(weight)
    print("End of  training")

    """
       Try to evaluate this trained model, on the test data.
       this will evaluate the accuracy of this model.
    """
    #score = model.evaluate(test_data, test_labels, verbose=isKerasVerbose)
    score = Evaluate(model, train_data, yTrainCat, xTest_bak, yTestCat)

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
        nn.updateModelJson(json_string)
    elif (nn.accuracy() > score[1]):
        GlobalTrainingTrend.updateDecInAccuracy(nn.accuracy(), score[1])
    else:
        print("")

    if EnableKerasDebug == True:
        print(history.losses)
        model.summary()
        #loss = history.losses
        #GlobalTrainingTrend.update(loss)


    KerasPlotModel(modelHistory, nn.nnName())


    # Actual accuracy calculated manually:

    return score[1]  # 1 is accuracy. 0 is loss.

def Evaluate(model, xTrain, yTrain, xTest, yTest):
    score = model.evaluate(xTest, yTest, verbose=isKerasVerbose)
    return score