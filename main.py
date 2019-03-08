import logging
from NNPorpulation import NeuralNetwork
from NNPorpulation import NNDb
from Params        import GlobalNNParamChoices as nnParams
from Params        import RandomChoiceFn as randomFn
from Params        import Population     as populationCount

from Algo          import TrainWithoutMutation as trainAndChooseTopAccuracy

from Algo          import TrainWithMutation   as trainWithEvolution

from Params        import involveMutation
#from tqdm import tqdm

from Params import LogFileName
from Params import ResultFileName
from Params import UpdateAllLogsToFile
import sys
fLog = open(LogFileName, 'w')

from Params import GlobalKnonwParams
def main():
    """Evolve a network."""
    generations = 10  # Number of times to evole the population.
    population = 20  # Number of networks in each generation.

    if UpdateAllLogsToFile is not None:
       sys.stdout = fLog

    print("Initializing with population:"+str(populationCount))

    db1 = NNDb(population - len(GlobalKnonwParams), randomFn, nnParams)

    for elem in GlobalKnonwParams:
        print("Adding with known params")
        db1.addEntryWithChoosenParam(elem)

    db1.createPopulation()
    print("Generated population:")



    pop = db1.population()

    print("Dumping by API")
    for nn in pop:
        nn.describe()
    print("End of Dumping")
    """
       Train network based on Neural Algorithms
    """
    trainedNNSet = []
    if involveMutation is not True:
       trainedNNSet = trainAndChooseTopAccuracy(db1)
    else:
       """
          Train network and then Evolve the population
       """
       trainedNNSet = trainWithEvolution(db1)

    for nn in trainedNNSet:
        nn.describe()
    fLog.close()


if __name__ == '__main__':
    main()

