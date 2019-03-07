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
from Params import DataSetFilePath

from Params import LoadPartial
from Params import NumOfFileToLoad

def load_data():
    path = "./ml_data/*.csv"
    file_name = ["./ml_data/1.csv", "./ml_data/2.csv", "./ml_data/3.csv", "./ml_data/4.csv",
                 "./ml_data/5.csv", "./ml_data/6.csv"]
    data = []
    labels = []

    print("loading data...")

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

    data = np.array(data)
    labels = np.array(labels)
    n = int(float(data.shape[0]) * 0.8)
    train_data = data[:n]
    train_labels = labels[:n]
    test_data = data[n:]
    test_labels = labels[n:]
    print("finished loading data")

    print(train_data)
    print(train_labels)
    return train_data, train_labels, test_data, test_labels

