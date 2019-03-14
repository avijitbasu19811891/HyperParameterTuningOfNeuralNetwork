"""
   Code this application
     Divided to 2 parts
      1. Generate module: Generate a population of NN, train and keep evelovving across generations
                          At each generation choose top few NN(fitness being accuracy on test data).
                          At end of each generation , add a few child, by choosing hyper params from 2 fit parents.
                          After a few few generation iteration, choose the top 1 and return
      2. Analyzer module: This will create a keras model. PArams of this keras will be the weight and json config of top
                          NN, saved at end of generate module.
                          Analyzer module will run, at interval, reading new data sets and predicting class of VM host.
                          Data being: VM host prams collected by monitoring of VM hosts. Thsi data collection is not part of this WS
                          Prediction: Class of this VM host 0 to 9
                          0-2 VM load will be light
                          3-6 Moderately loaded
                          7-9 Highly loaded VM

                          Predicted label will be used by a seperate Sw to take corrective actions like
                          1. Rebablance Sheaf
                          2. Rebalance VM hosts.
                          3. Shutdown Highly loaded VMs.
"""
from GenerateModule import GenerateAndTrainNN

from Params import GlobalSwMode
from LaunchTrainedNN import AnalyzerModule


from Params import NeuronConfigFile
from Params import NeuronWeightFile

import os
from Params import LogFileName
from Params import ResultFileName
from Params import UpdateAllLogsToFile
import sys

"""
Deal with various file related operations 
  Files are
    Log file
    Result file to capture selected NN.
       This will will help to train NN and then resintantiate from the result.
"""
class FileModules:
    def __init__(self):
        self._oldStout = None
        self._logFile  = None
        self._fResultConf = None
        self._fWeightFile = None
        """
        try:
            os.mkdir("../Result")
        except OSError:
            print ("Creation of the directory failed")
        else:
            print ("Successfully created the directory")
        """

    def redirctPrinttoFile(self, fileName =None):
        if fileName is not None:
            self._logFile = open(fileName, 'w')
        else:
            self._logFile = open(LogFileName, "w")

        self._oldStout = sys.stdout
        sys.stdout = self._logFile
    def closePrint(self):
        sys.stdout = self._oldStout
        if self._logFile  is not None:
            self._logFile.close()
    def getResultFile(self):
        self._fResultConf = open(NeuronConfigFile, "w")
        self._fWeightFile = open(NeuronWeightFile, "w")
        return self._fWeightFile, self._fResultConf

    def onCloseResultFile(self):
        if self._fWeightFile is not None:
            self._fWeightFile.close()
        if self._fResultConf is not None:
            self._fResultConf.close()
    def readNeuronResult(self):
        self._fResultConf = open(NeuronConfigFile, "r")
        self._fWeightFile = open(NeuronWeightFile, "r")
        return self._fWeightFile, self._fResultConf

GlobalfileModule = FileModules()

from KerasModule import SaveResult
from KerasModule import LoadResult

from Analyzer import AnalyzerThread

import time
from DataSet import PrintDataTrend
from DataSet import load_prediction_data as MaintainceLog


def launchSw ():

    optimizer = 'rmsprop'
    """
       Redirect all prints to a log file
    """
    if UpdateAllLogsToFile is not None:
        GlobalfileModule.redirctPrinttoFile()

    """
       Train networks
       This will result the json-conf and weight of the top performing NN
    """
    if GlobalSwMode == 'Train':
        weightArray , jsonConf, numLayers, numNeurons, activation, optimizer = GenerateAndTrainNN(NeuronConfigFile, NeuronWeightFile)

        """
           Save to file
         """
        fWeight , fConf = GlobalfileModule.getResultFile()
        SaveResult(fConf = fConf, fWeight = fWeight, config=jsonConf, weight=weightArray)
        GlobalfileModule.onCloseResultFile()

    fRdWeight, fRdConfig = GlobalfileModule.readNeuronResult()

    NNConfig, NNWeight = LoadResult(fWeight = fRdWeight, fConf= fRdConfig)

    print("Analyzer not yet supported")

    """
      Read the saved config , weight and recreate network
    """

    analyzer = AnalyzerModule(isConfigFromFile=False,
                              jsonConfig = NNConfig,
                              weightConfig = NNWeight,
                              jsonCnfFile = None,
                              weightCnfFile = None,
                              numLayers = None,
                              numNeurons = None,
                              activation = None,
                              optimizer = optimizer
                              )
    analyzer.describe()

    GlobalfileModule.closePrint()

    """
       Periodically read csv files for 2 host.
       invoke analyzer.run() and update to a file(file name indexed by vmm name) 
    """
    analyzerThread = AnalyzerThread(Predictor=analyzer)

    analyzerThread.run()

    GlobalfileModule.closePrint()