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
        self._numOfTimesAccInc = 0
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
        self._numOfTimesAccInc += 1
        self._accuracyIncTrend.append([old,new])


    def describe(self):
        print("Num of times accucracy improved:" + str(self._numOfTimesAccInc))
        print("Num of times accucracy dec:"+str(self._numOfTimesAccDec))
        for elem in self._accuracyDecTrend:
            print(elem)

        plt.plot(self._avgFitness)
        plt.title('Avg Fitness(Accuracy)')
        plt.ylabel('Accuracy')
        plt.xlabel('Genrration')
        plt.savefig(GraphPath + "/" + "Fitness" + '.png')

        plt.close()

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

from Params import GraphPath

FileNumber = 0
def KerasPlotModel(history, caption=None):
    global FileNumber
    """
    # Plot training & validation accuracy values

    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.savefig(GraphPath+ "/" + "Training"+str(FileNumber)+"loss"+'.png')
    """
    FileNumber = (int)(FileNumber+1)
    # Plot training & validation loss values
    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title("Model Accuracy for"+caption)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(GraphPath + "/" + "Training"+str(FileNumber) + "Accuracy" + '.png')
    plt.close()

def PlotIndividualTrainingTrend(NNName, accuracy):
    global FileNumber
    FileNumber = (int)(FileNumber+1)
    # Plot training & validation loss values
    plt.plot(accuracy, 'ro')
    plt.title("Neural Network accuracy Trend for"+NNName)
    plt.ylabel('accuracy')
    plt.xlabel('Iteration')
    plt.savefig(GraphPath + "/" + NNName+"Training"+str(FileNumber) + "Loss" + '.png')
    plt.close()

def PicturizeTestSet(model,xTest, yTest):
    predicted_classes = model.predict_classes(xTrain)
    correct_indices = np.nonzero(predicted_classes == yTest)[0]
    incorrect_indices = np.nonzero(predicted_classes != yTest)[0]
    print("in correct prediction")
    print("Predicted, Correct")

from sklearn.metrics import classification_report, confusion_matrix
def PrintConfusionMatrix(xTest, yTest, yPred):
    return
    #matrix = confusion_matrix(yTest, yPred)
    #print(matrix)




GlobalTrainingTrend = KegarTrainingTrend()