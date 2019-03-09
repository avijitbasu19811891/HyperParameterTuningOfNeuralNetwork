""" Training of population
    this is the place holder for the NN training logic
"""
import random
from NNPorpulation import NeuralNetwork as neuralNet

#from Trainer import arnn as train
from Trainer import trainKeras as train
def TrainAndScore(nn, train_data, train_labels, test_data=None, test_labels=None):
    print("Training")
    """ Initiate training Algo on this nn
        Retrieve and update Accuracy
    """
    nn.describe()
    score = train(nn, train_data, train_labels, test_data, test_labels)
    print("Completed Training of network")
    nn.describe()
