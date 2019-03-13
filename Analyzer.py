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

"""
def confusionMatrix(yPred, yExpect):
    compare = yPred == yExpect

    position = 0
    ErrorCount = 0
    CorrectCount = 0
    for c in compare:
        if c == False:
           ErrorCount += 1
        else:
            CorrectCount += 1
    return [ErrorCount, CorrectCount]

"""
from sklearn.metrics import classification_report, confusion_matrix

def confusionMatrix(yPred, yExpect, fileName):
    cm = confusion_matrix(yExpect, yPred)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(333)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    #ax.set_xticklabels([''] + labels)
    #ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(fileName)
    plt.close()




def DumpData(data, predLabels, labels=None):
    global PredictFileNumber
    PredictFileNumber = (int)(PredictFileNumber + 1)

    shape = [data.shape[1], data.shape[0]]


    dataToPlot = np.reshape(data, shape)

    dataToPlot = dataToPlot[:3]


    """
    plt.plot(dataToPlot)
    plt.plot(predLabels)
    plt.title("Prediction")
    plt.ylabel('Values')
    plt.xlabel('Time')
    #plt.legend(['Train', 'Test'], loc='upper left')
    """
    fig = plt.figure()
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)


    ax1.plot( dataToPlot)
    ax2.plot( predLabels)


    ax1.get_shared_x_axes().join(ax1, ax2)
    ax1.set_xticklabels([])

    ax1.yaxis.set_label("Vm CPU, Mem, I/O load")
    ax1.yaxis.set_label("Predicted Class")
    plt.savefig(PredictResultPath +"Predict" +str(PredictFileNumber)+ '.png')
    plt.close()
    fname = PredictResultPath +"Confusion" +str(PredictFileNumber)+ '.png'
    if labels is not None:
        confusionMatrix(yPred=predLabels, yExpect=labels, fileName=fname)
        #print("PredictionStats Error"+str(err) +"vs Expected:"+ str(correct))


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

        DumpData(data=testData, predLabels=predResult, labels=expectLabel)

        print("")
    def run(self):
        """ Method that runs forever """
        while True:
            # Do something
            print('Doing something imporant in the background')

            self.loadData()
            time.sleep(self.interval)


