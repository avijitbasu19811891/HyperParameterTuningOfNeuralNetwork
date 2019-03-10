
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
            self._model = keras.models.Sequential()

    def executePrediction(self, Data):
        predictResult = None
        if self._model is not None:
            predictResult = self._model.predict(data)
        return predictResult

    def describe(self):
        self._model.summary()
    def plot(self):
        if self._model is not None:
            plot_model(self._model, to_file='../Graphs/model_plot.png', show_shapes=True, show_layer_names=True)


"""
  Save result of a model to file.
  This will help relaunch of NN, without training on every run of SW
"""
def SaveResult( fConf, fWeight,config= None, weight= None):
    if config is None:
        fConf.write("Result not Updated")
    if weight is None:
        fWeight.write("Wwight not updated")
    else:
        print(weight)
        #np.savetxt("saved_numpy_data.csv", weight, delimiter=",", format='%10.5f')

