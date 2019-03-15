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

def ClassToDescription(c):

    if c < 2:
        desc = "VM Lightly Loaded \n"
    elif c < 5:
        desc = "Vm load is moderately higher \n"
    elif c < 8:
        desc = "Vm load on Higher side \n"
    else:
        desc = "VM in Host zone, need to be migrated"
    return "Class_"+str(c)+"_," + "_"+ desc

def confusionMatrix(yPred, yExpect, fileName):
    cm = confusion_matrix(yExpect, yPred)
    print(cm)
    labels=['0','1','2','3','4','5','6','7','8','9']
    plotMatrix(cm, labels, "ConfusionMatrix of Prediction", fileName)

import pandas as pd

def dataDescription(data, labels):
    # Creating pandas dataframe from numpy array
    data = np.asarray(data)
    labels = np.asarray(labels)

    descr = []
    for p in labels:
        descr.append(ClassToDescription(p))

    descr = np.asarray(descr)
    print(descr.shape)

    dataset = pd.DataFrame({'CPU load': data[:, 0],
                            'Memory:': data[:, 1],
                            'I/O:': data[:, 2],
                            'Network:': data[:, 3],
                            'Params1:': data[:,4],
                            'Params2:' : data[:,5],
                            'Params3:' : data[:,6],
                            'Params4:' : data[:,7],
                            'Params5:' : data[:,8],
                            'VM_LOAD_LEVEL:': labels,
                            'descr:': descr})

    print(dataset)
    return dataset

import datetime
def UpdatePredictionResult(data, predLabels, labels=None):
    dataSet = dataDescription(data, predLabels)

    fname = PredictResultPath + "VmHost_Analyzer.csv"

    dataSet.to_csv(fname, index = None,mode='a', header=['CPU load', 'Memory:', 'I/O:', 'Network',
                                                         'Params1:', 'Params2:','Params3:',
                                                         'Params4:', 'Params5:',
                                                         'VM_LOAD_LEVEL:', 'descr:'])

    """
    fname = PredictResultPath + "VmHost_Analyzer.txt"

    idx = 0
    f = open(fname, "a")
    now = datetime.datetime.now()
    f.write("\n \n \n Storing VM host load and suggested action on "+str(now)+"\n")
    for d in data:
       d = np.asarray(d)

       f.write("\n Host Monitoring Data:\n")
       np.savetxt(f, d)
       f.write("\n"+ClassToDescription(predLabels[idx])+"\n \n")
       idx += 1
    f.write("\n \n \n")
    f.close()
    """

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

    def __init__(self, interval=40, callBack =None, Args =None, Predictor = None):
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


