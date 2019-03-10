

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

"""
Class to launch a Keras from an existing jsonc and predefined weights.
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
            self._model = keras.models.Sequential(configJson)
            if weight is not None:
               self._model.set_weights(weight)
        else:
            print("Not supported to read result from file")

    def executePrediction(self, Data):
        predictResult = None
        if self._model is not None:
            predictResult = self._model.predict(data)
        return predictResult



def SaveResult( fConf, fWeight,config= None, weight= None):
    if config is None:
        fConf.write("Result not Updated")
    if weight is None:
        fWeight.write("Wwight not updated")

