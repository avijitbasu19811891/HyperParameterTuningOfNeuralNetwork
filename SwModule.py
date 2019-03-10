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
        return self._fResultConf, self._fWeightFile

    def onCloseResultFile(self):
        if self._fWeightFile is not None:
            self._fWeightFile.close()
        if self._fResultConf is not None:
            self._fResultConf.close()
    def readNeuronResult(self):
        self._fResultConf = open(NeuronConfigFile, "r")
        self._fWeightFile = open(NeuronWeightFile, "r")
        return self._fResultConf, self._fWeightFile

GlobalfileModule = FileModules()

from KerasModule import SaveResult
def launchSw ():

    if UpdateAllLogsToFile is not None:
        GlobalfileModule.redirctPrinttoFile()

    """
       Train networks
       This will result the json-conf and weight of the top performing NN
    """
    if GlobalSwMode == 'Train':
        jsonConf, weightArray = GenerateAndTrainNN(NeuronConfigFile, NeuronWeightFile)

    """
       Save to file
    """
    fConf, fWeight = GlobalfileModule.getResultFile()
    SaveResult(fConf, fWeight)
    GlobalfileModule.onCloseResultFile()

    GlobalfileModule.closePrint()
    print("Analyzer not yet supported")

    """
      Read the saved config , weight and recreate network
    """

    analyzer = AnalyzerModule(isConfigFromFile=False,
                              jsonConfig = jsonConf,
                              weightConfig = weightArray,
                              jsonCnfFile = None,
                              weightCnfFile = None
                              )
    analyzer.describe()

    GlobalfileModule.closePrint()
