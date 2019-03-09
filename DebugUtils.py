from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class KegarTrainingTrend:
    def __init__(self):
        self._lossTrend = []
        self._avgFitness = []
        self._numOfTimesAccDec = 0
        self._accuracyDecTrend = []
        self._accuracyIncTrend = 0
        self._accuracyIncTrend = []
        self._topFitness = []
    def update (self,loss):
        self._lossTrend.append(loss)
    def updateavgFitness (self,fitness):
        self._avgFitness.append(fitness)
    def updateTopAccuracy(self, top):
        self.topFitness.append(top)

    def updateDecInAccuracy(self, old, new):
        self._numOfTimesAccDec += 1
        self._accuracyDecTrend.append([old,new])
    def updateIncInAccuracy(self, old, new):
        self._accuracyIncTrend += 1
        self._accuracyIncTrend.append([old,new])


    def describe(self):
        print("Num of times accucracy dec:"+str(self._numOfTimesAccDec))
        for elem in self._accuracyDecTrend:
            print(elem)
            """
            fig, (ax1) = plt.subplots()

            ax1.set_title('Progress of training')
            ax1.semilogy(self._avgFitness)
            ax1.set_ylabel('Avg Fitness')
            ax1.grid(True, which="both")
            ax1.margins(0, 0.05)

            """
            plt.plot(self._avgFitness)
            plt.tight_layout()
            plt.show()


    """
        isImproving = 1
        for lossPerTraining in self._lossTrend:
            lastGood = None
            for loss in lossPerTraining:
                if lastGood is not None:
                    if loss > lastGood:
                        isImproving = 0
                else:
                    if loss < lastGood:
                        lastGood = loss
            print("Did Training improve:"+str(isImproving))
            print(lossPerTraining)
    """

GlobalTrainingTrend = KegarTrainingTrend()