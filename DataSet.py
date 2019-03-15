"""
Place holder for
   extraction of data set
   preprocessing of the Data set
"""
import glob
import csv
import numpy as np

sample_history = 10

GlobalDataSet = []
from Params import DataSetFilePath as path

from Params import LoadPartial
from Params import NumOfFileToLoad
#import seaborn as sns
#from scipy import stats
#import numpy as np
from scipy import stats


def evaluateZscore(x):
    withinZscore = True

    for elem in x[:-1]:
        if elem < 3:
            continue
        else:
            withinZscore = False
            break
    return withinZscore

def removeOutlier(data, label):
    retData = []
    retlabel = []

    z = np.abs(stats.zscore(data))

    z = ((z < 3).all(axis=1))

    numEliminated = 0
    numIncluded = 0
    print(z)
    print(len(z))
    for idx in range(len(z)):
        if z[idx] == False:
            numEliminated += 1
        else:
            retData.append(data[idx])
            retlabel.append(label[idx])
            numIncluded += 1

    print("Preprocessing concluded:")
    print("Excluded:"+str(numEliminated)+"From:"+str(idx)+"DataSet")
    return retData, retlabel



from Params import RemoveOutlier

def TuneLabels(labels):
    for idx in range(len(labels)):
        if labels[idx] > 5.0:
            labels[idx] = 1.0
        else:
            labels[idx] = 0.0
    return labels


def PrintDataTrend (data, labels, caption):
    unique, counts = np.unique(labels, return_counts=True)
    print(caption+" Data count:"+str(len(labels)))
    print np.asarray((unique, counts)).T

import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

"""
   @brief Provide some formatting and some visualtization of data being used for
           training and for testing
"""
def FormatData(data, labels):
    headerInfo = ['CPU load', 'Memory:', 'I/O:', 'Network:', 'Params1:',
                  'Params2:', 'Params3:', 'Params4:', 'Params5:', 'Label']
    dataSet = pd.DataFrame({'CPU load': data[:, 0],
                            'Memory:': data[:, 1],
                            'I/O:': data[:, 2],
                            'Network:': data[:, 3],
                            'Params1:': data[:,4],
                            'Params2:' : data[:,5],
                            'Params3:' : data[:,6],
                            'Params4:' : data[:,7],
                            'Params5:' : data[:,8],
                            'VM_LOAD_LEVEL:': labels})
    dataSet.to_csv("../FormattedData/FormattedData.csv", index=True,
                   index_label='Time Instance',
                   header=headerInfo)

    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(121)

    ax1.set_title('with log scale')
    dataSet.hist(column=['CPU load', 'Memory:', 'I/O:', 'Params5:','VM_LOAD_LEVEL:'], ax=ax1, bins=4)
    ax1.set_yscale('log')
    fig.savefig("../Graphs/TrainingData_hist.png", dpi=240)
    plt.close()

    correlations = dataSet.corr()
    # plot correlation matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations)
    fig.colorbar(cax)
    """
    ticks = numpy.arange(0, 9, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    """
    ax.set_xticklabels(headerInfo)
    ax.set_yticklabels(headerInfo)
    plt.savefig("../Graphs/TrainingData_corr.png")
    plt.close()

    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(121)
    dataSet.plot(kind='density', subplots=True, layout=(4, 3), sharex=False, ax=ax1)
    plt.savefig("../Graphs/TrainingData_density.png")
    plt.close()


def load_data():
    data = []
    labels = []

    print("loading data...")
    finalToUse = []
    idx = 0

    for fname in glob.glob(path):
        print (fname)
        with open(fname, 'r') as infh:
            reader = csv.reader(infh, delimiter=';')

            for row in reader:
                r = np.array(row, dtype=float)
                rr = []
                for i in range(sample_history):
                    rr.append(r[i * 7 + 1])
                # print(rr)
                data.append(rr)
                labels.append(r[-1])
        if LoadPartial is not None:
            if idx > NumOfFileToLoad:
                break
        idx +=1

    if RemoveOutlier == True:
        data, labels = removeOutlier(data, labels)

    data = np.array(data)
    labels = np.array(labels)

    FormatData(data, labels)
    n = int(float(data.shape[0]) * 0.8)
    train_data = data[:n]
    train_labels = labels[:n]
    test_data = data[n:]
    test_labels = labels[n:]
    print("finished loading data")

    print(train_data)
    print(train_labels)
    #test_data = removeOutlier(data)

    PrintDataTrend(train_data, train_labels,"Training Data Stats")
    PrintDataTrend(test_data, test_labels, "Testing Accuracy on Data Stats")
    return train_data, train_labels, test_data, test_labels


import os
import random

def load_prediction_data():
    maintainceData = []
    # make random
    data = []
    labels = []
    flist = []
    print("loading data for prediction...")
    i = 0
    idx = 0
    for fname in glob.glob(path):
        fileName = "../ml_data/"+fname
        maintainceData.append(fileName)
        if LoadPartial is not None:
            if idx > NumOfFileToLoad:
                break
        idx +=1

    for idx in range(0,7):
        fname = random.choice(maintainceData)
        flist.append(fname)

    for fname in flist:
        #fname = "../ml_data/"+fname
        print("Loading"+fname)
        with open(fname, 'r') as infh:
            reader = csv.reader(infh, delimiter=';')
            for row in reader:
               r = np.array(row, dtype=float)
               rr = []
               for i in range(sample_history):
                   rr.append(r[i * 7 + 1])
                   # print(rr)
                   data.append(rr)
                   labels.append(r[-1])
                   i = i+1

    data = np.array(data)
    labels = np.array(labels)

    n = int(float(data.shape[0]) * 0.8)
    test_data = data[:n]
    test_labels = labels[:n]

    return test_data, test_labels

