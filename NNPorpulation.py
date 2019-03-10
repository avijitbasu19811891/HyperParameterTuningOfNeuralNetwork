import numpy as np
import random
import tensorflow as tf

def testPrint(text):
    print(text)

class NeuralNetwork:
    """Represent a network and let us operate on it.
    Currently only works for an MLP.
    """

    def __init__(self, nnParams, randomFn, nn_param_choices=None):
        """Initialize our network.
        Args:
            nn_param_choices (dict): Parameters for the network, includes:
                nb_neurons (list): [64, 128, 256]
                nb_layers (list): [1, 2, 3, 4]
                activation (list): ['relu', 'elu']
                optimizer (list): ['rmsprop', 'adam']
        """
        self._accuracy = 0.
        if nnParams is None:
            self._nn_param_choices = nn_param_choices
            self._network = {}  # (dic): represents MLP network parameters
            self._ramdonFn = randomFn
            """Create a random network."""
            for key in self._nn_param_choices:
                print("Choosing params")
                print(self._nn_param_choices[key])
                self._network[key] = random.choice(self._nn_param_choices[key])
        else:
            self._network = nnParams

        self._weight = []
        self._trainingModel = None
        self._summary = None
        self._lossTrend = []
        self._accuracyTrend = []
        self._jsonConf = None
        self._accuracyTrend.append(0.0)

    def nnName(self):
        name = "NN_"+str(self._network['nb_neurons'])+"_Neurons_"+str(self._network['nb_layers'])+"_Layers_"+self._network['activation']+"_act"
        return name
    def updateAccuracy(self, accuracy):
        print("Updating accuracy")
        self._accuracy = accuracy

    def accuracy(self):
        return self._accuracy

    def updateWeight(self, weight):
        self._weight = weight

    def isWeightSet(self):
        return len(self._weight)

    def getWeight(self):
        return self._weight

    def updateModel(self, model):
        self._trainingModel = model
    def updateModelJson(self, config):
        self._jsonConf = config

    def getModelDetails(self):
        return self._jsonConf

    def numLayers(self):
        return self._network['nb_layers']

    def numNeurons(self):
        return self._network['nb_neurons']

    def activation(self):
        return self._network['activation']
    def optimizer(self):
        return self._network['optimizer']

    def getParams(self):
        return self.numLayers(), self.numNeurons(), self.activation(), self.optimizer()

    def updateSummary(self, summary):
        self._summary = summary

    """
       Update entire set of neural parameters.
    """
    def updateParams(self,nn_params):
        self._network = nn_params

    """
       Update one of neural parameters.
    """
    def updateParams(self,key, param):
       self._network[key] = param

    def accuracy(self):
        return self._accuracy

    def updateTrainingHistory(self, accuracy, loss = None):
        self._accuracyTrend.append(accuracy)
        if loss is not None:
            self._lossTrend.append(loss)


    def trainingTrend(self):
        return self._accuracyTrend

    def describe(self, detailed = False, fileId=None):
        if fileId is None:
           print(self._network)
           print("Accuracy:"+str(self._accuracy))
           if self._summary is not None:
               print(self._summary)
           if detailed is True:
              print("Weight:")
              print(self._weight)
        else:
            print(self._network,fileId)
            print("Accuracy:"+str(self._accuracy),fileId)



class NNDb:
    def __init__(self, population, randomFn, nn_param_choices=None):
        self._set = []
        self._numNN = 0
        self._population = population
        self._paramChoice = nn_param_choices
        self._randomFn = randomFn

    def addEntry(self):
        nn = NeuralNetwork(None, self._randomFn, self._paramChoice)
        nn.describe()
        self._set.append(nn)
        self._numNN += 1

    def addEntryWithChoosenParam(self, params):
        print("Adding known networks")
        print(params)
        nn = NeuralNetwork(params, None)
        self._set.append(nn)
        self._numNN += 1

    def createPopulation(self):
        for idx in range(1 , self._population):
            self.addEntry()

    def runOnPopulation(self, fn=None):
        for nn in self._set:
            if (fn is not None):
                fn(nn)
    def population(self):
        pop = []
        for nn in self._set:
            pop.append(nn)
        return pop

    def describe(self):
        print("NNDB:")
        print("Params:")
        print(self._paramChoice)
        print("NumNN:"+str(self._numNN))
        for nn in self._set:
            nn.describe()


if int((tf.__version__).split('.')[1]) < 12:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()

