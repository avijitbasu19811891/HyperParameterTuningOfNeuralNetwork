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
    #analyzerThread = AnalyzerThread()



    testData, expectLabel = MaintainceLog()

    print(testData)
    print("Expected labels")
    print(expectLabel)

    print("Predicting")
    predResult = analyzer.run(testData)

    print(predResult)

    print("")

    GlobalfileModule.closePrint()