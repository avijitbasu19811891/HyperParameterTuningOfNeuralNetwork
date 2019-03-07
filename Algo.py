"""
    Place holder for algo that trains and impoves accracy of the NN
    and chooses the best NN
"""
import logging
from NNPorpulation import NeuralNetwork
from NNPorpulation import NNDb
from Params        import GlobalNNParamChoices as nnParams
from Params        import RandomChoiceFn as randomFn
from NNTraining   import TrainAndScore as trainNN
from GenticOnNNPopulation import TriggerNNEvolution as evolve
from Params        import Population     as populationCount
from Params        import Generations    as generationCount
from Params        import TopBreedPercent as topPercent
from GenticOnNNPopulation import  MutateNN as mutateNN

def printDeveloppedNN(nnSet, caption):
    print("")
    print("============================================================")
    print(caption)
    for idx in range(0 , len(nnSet)):
        nnSet[idx].describe()
    print("============================================================")
    print("")

"""
Train a given NN.
This will update the weight of the NN and produce an accuracy measurement
"""
from DataSet import load_data as ld
train_data, train_labels, test_data, test_labels = ld()

def TriggerTraining(nn, trainingfn=None):
    print("Triggering training for network:")
    trainNN(nn, train_data, train_labels, test_data, test_labels)


from NNTraining   import TrainAndScore as trainNN
from Algo          import  TriggerTraining as trainAndEstimate
from GenticOnNNPopulation import TriggerNNEvolution as evolve

"""
This is the algorithm that trains and chooses top few NN 
"""
def TrainWithoutMutation(db, dataSet=None):
    trainingTrigger = lambda nn: trainAndEstimate(nn)

    db.runOnPopulation(trainingTrigger)

    print("Db after 1st round of training")
    printNN = lambda nn: nn.describe()
    db.runOnPopulation(printNN)

    """
    Sort based on accuracy the first  5% of the population
    """
    numNNToChoose = int((float(populationCount)*topPercent)/100)

    print("Choose top:"+str(numNNToChoose))

    fitness = lambda nn: nn.accuracy()

    gradedNN = [(fitness(nn), nn) for nn in db._set]

    gradedNN = [x[1] for x in sorted(gradedNN, key=lambda x: x[0], reverse=True)]
    print("Sorted on Accuracy")
    print(gradedNN)


    topNN = gradedNN[:numNNToChoose]
    topNN[1].describe()
    print("Describing top NN choosen after training")
    printDeveloppedNN(topNN, "Describing top NN choosen after training")

    return topNN

"""
   This works on a population of NN.
   This works on a set of generations of the population and evelove each population
   Then after each evolution, train the population of evolved NN
"""
def TrainGeneration(population, dataSet=None):
    trainPopulation = lambda nn: trainAndEstimate(nn)

    fitness = lambda nn: nn.accuracy()

    """
       Run though each generation of population set
    """
    for idx in range(1, generationCount):
        """
           evlove this poluation set
        """
        evolvedPopulation = evolve(population, fitness)
        idx += 1
        """
           Train this set of neural networks
        """
        for nn in evolvedPopulation:
            trainPopulation(nn)

        """
           <TBD> We need to make sure, that this evolvedPopulation is
           better than earlier populaiton
        """
    return evolvedPopulation

from Algo import TrainGeneration as generationFn

"""
This is the algorithm that trains and eveloves the set of NN
"""
def TrainWithMutation(db, dataSet=None):
    trainingTrigger = lambda nn: trainAndEstimate(nn)

    db.runOnPopulation(trainingTrigger)

    print("Db after 1st round of training")
    printNN = lambda nn: nn.describe()
    db.runOnPopulation(printNN)
    developpedNNSet = generationFn(db.population())
    """
    fitness = lambda nn: nn.accuracy()
    developpedNNSet = evolve(db.population(), fitness)
    printDeveloppedNN(developpedNNSet, "Describing population after Evolution")
    """


    """
    Sort based on accuracy the first  5% of the population
    """
    numNNToChoose = int((float(populationCount)*topPercent)/100)

    print("Choose top:"+str(numNNToChoose))

    topNN = developpedNNSet[:numNNToChoose]

    return topNN