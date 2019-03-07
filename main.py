import logging
from NNPorpulation import NeuralNetwork
from NNPorpulation import NNDb
from Params        import GlobalNNParamChoices as nnParams
from Params        import RandomChoiceFn as randomFn
from Params        import Population     as populationCount

from Algo          import TrainWithoutMutation as trainAndChooseTopAccuracy

from Algo          import TrainWithMutation   as trainWithEvolution

#from tqdm import tqdm

def main():
    """Evolve a network."""
    generations = 10  # Number of times to evole the population.
    population = 20  # Number of networks in each generation.

    print("Initializing with population:"+str(populationCount))

    db1 = NNDb(population, randomFn, nnParams)

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
    #trainedNNSet = trainAndChooseTopAccuracy(db1)

    """
       Train network and then Evolve the population
    """

    traninedAndEvolvedNNSet = trainWithEvolution(db1)

if __name__ == '__main__':
    main()

