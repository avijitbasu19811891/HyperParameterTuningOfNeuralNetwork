
"""
   Code to host a Keras model that is spawned after training.
      This model will be mostly host from set of parameters saved after Training Phase of networks
      are complete
"""
import numpy as np
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



from DebugUtils import GlobalTrainingTrend

from DebugUtils import KerasPlotModel

from DebugUtils import PrintConfusionMatrix

from keras.utils.vis_utils import plot_model
"""
Class to launch a Keras from an existing jsonconf and predefined weights.
"""
class KerasModel:
    """
       @brief Model can be launched from previously saved weight and config
       @param[in]  weight  numpy array of weights
       @param[in]  configJson Json string for config of this model
    """
    def __init__(self, weight,
                 configJson,
                 numLayers,
                 numNeurons,
                 activation,
                 optimizer
                 ):
        self._model = None
        if configJson is not None:
            print(" Initiating Model ->")
            self._model = keras.models.model_from_json(configJson)
            if weight is not None:
               self._model.set_weights(weight)
        elif numLayers is None:
            print("Not supported to read result from file")
        else:
            """ 
               Generate model from params
            """
            print("Not supported")

    def train(self, data, labels):
        history = self._model.fit(data, labels,
              batch_size=10,
              epochs=10,  # using early stopping, so no real limit
              verbose=2,
              validation_data=(data, labels),
              callbacks = None)

              #callbacks= EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto', baseline=None, restore_best_weights=False))


        score = self._model.evaluate(data, labels, verbose=2)
        print("Scores after training")
        print(score)

    def executePrediction(self, data):
        predictResult = None
        if self._model is not None:
            predictResult = self._model.predict_classes(data)
        return predictResult

    def describe(self):
        self._model.summary()
    def plot(self):
        if self._model is not None:
            plot_model(self._model, to_file='../Graphs/model_plot.png', show_shapes=True, show_layer_names=True)



import pickle
"""
  Save result of a model to file.
  This will help relaunch of NN, without training on every run of SW
"""
def SaveResult( fConf, fWeight,config= None, weight= None):
    if config is None:
        fConf.write("Config not updated")
    else:
        fConf.write(config)
    if weight is None:
        fWeight.write("Weight not updated")
    else:
        print("Saving weight")
        print(weight)
        pickle.dump(weight, fWeight)


"""
  Save result of a model to file.
  This will help relaunch of NN, without training on every run of SW
"""

def LoadResult( fConf, fWeight):
    config = None
    weight = None
    if fConf is None:
        print("Config file not specified")
    else:
        config = fConf.read()
        print("Reading config")
        print(config)

    if fWeight is None:
        print("Weight file not specified")
    else:
        weight = pickle.load(fWeight)
        print(" Read weight")
        print(weight)
    return config, weight
