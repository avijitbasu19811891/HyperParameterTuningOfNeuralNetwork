from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

import sys

from  KerasModule import  KerasModel

class AnalyzerModule:
    def __init__(self, isConfigFromFile, jsonConfig, weightConfig,
                 jsonCnfFile, weightCnfFile):
        self._kerasModel = None
        if isConfigFromFile == True:
           return
        else:
            self._kerasModel = KerasModel(weight= weightConfig,
                                          configJson = jsonConfig,
                                          numLayers =None,
                                          numNeurons = None,
                                          activation = None,
                                          optimizer = None);


        if self._kerasModel is not None:
            print("Trained NNN launched -> start predicting")

    def run(self, DataSet):
        print(" Predicting")
        result = self._kerasModel.executePrediction()

    def describe(self):
        if self._kerasModel is not None:
            print(self._kerasModel.to_json())



