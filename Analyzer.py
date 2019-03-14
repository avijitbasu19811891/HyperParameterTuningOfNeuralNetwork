import threading
import time

import numpy as np
from DataSet import load_prediction_data as MaintainceLog
#from DataSet import DumpData
from LaunchTrainedNN import AnalyzerModule

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pandas as pd

PredictResultPath = "../PredictResult/"
PredictFileNumber = 0


from PlotGraphs import plotMatrix
from PlotGraphs import PlotData
from sklearn.metrics import classification_report, confusion_matrix

def confusionMatrix(yPred, yExpect, fileName):
    cm = confusion_matrix(yExpect, yPred)
    print(cm)
    labels=['0','1','2','3','4','5','6','7','8','9']
    plotMatrix(cm, labels, "ConfusionMatrix of Prediction", fileName)


def UpdatePredictionResult(data, predLabels, labels=None):
    global PredictFileNumber
    PredictFileNumber = (int)(PredictFileNumber + 1)

    fname = PredictResultPath +"PredictionResult" +str(PredictFileNumber)+ '.png'
    PlotData(data=data, labels=predLabels, fileInfo=fname, Caption="PredictionResult")
    if labels is not None:
        fname = PredictResultPath + "PredictionConfusionMat" + str(PredictFileNumber) + '.png'
        confusionMatrix(yPred=predLabels, yExpect=labels, fileName=fname)


class AnalyzerThread():
    """ Threading example class
    The run() method will be started and it will run in the background
    until the application exits.
    """

    def __init__(self, interval=20, callBack =None, Args =None, Predictor = None):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """
        self.interval = interval
        self._predictor = Predictor
    """
        if callBack is None:
            thread = threading.Thread(target=self.run, args=())
        else:
            thread = threading.Thread(target=callBack, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution

    """
    def loadData(self):
        testData, expectLabel = MaintainceLog()


        print("Expected labels")
        print(expectLabel)

        print("Predicting")
        predResult = self._predictor.run(testData)

        print(predResult)

        UpdatePredictionResult(data=testData, predLabels=predResult, labels=expectLabel)

        print("")
    def run(self):
        """ Method that runs forever """
        while True:
            # Do something
            print('Doing something imporant in the background')

            self.loadData()
            time.sleep(self.interval)


