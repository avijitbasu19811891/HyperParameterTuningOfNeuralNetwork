"""
    Code to launch a Analyzer model, from the last trained parameters.
    This Analyzer will perform following
        1. Launch a keras model from a numpy array of weights, json string of config
           Save this model.
        2. Provide a run() function to take a dataset and execute prediction
           This passes the params to the keras model launched earlier

"""
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

import sys

from  KerasModule import  KerasModel

class AnalyzerModule:
    def __init__(self, isConfigFromFile, jsonConfig, weightConfig,
                 jsonCnfFile, weightCnfFile,
                 numLayers,
                 numNeurons,
                 activation,
                 optimizer):
        self._kerasModel = None
        if isConfigFromFile == True:
           print("Read from Config file not supported")
           return
        else:
            self._kerasModel = KerasModel(weight= weightConfig,
                                          configJson = jsonConfig,
                                          numLayers =numLayers,
                                          numNeurons = numNeurons,
                                          activation = activation,
                                          optimizer = optimizer);


        if self._kerasModel is not None:
            print("Trained NNN launched -> start predicting")

    def run(self, DataSet):
        print(" Predicting")
        result = self._kerasModel.executePrediction(DataSet)
        return result

    def train(self, data, labels):
        return self._kerasModel.train( data, labels)

    def describe(self):
        if self._kerasModel is not None:
            self._kerasModel.describe()



