GlobalNNParamChoices = {
    'nb_neurons': [64, 128, 256, 512],
    'nb_layers': [1, 2, 3, 5, 7],
    'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
    'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                  'adadelta', 'adamax', 'nadam'],
}

def RandomChoiceFn (key, paramsChoice):
    return random.choice(paramsChoice[key])

# Number of times to evolve the population.
Generations = 3
# Number of networks in population
Population = 20

"""
From each generation choose the top
 few as the fittest.
"""
TopBreedPercent = 15
FitnessPopulationPercent = 70

"""
  Details of location of  Data set
"""
DataSetFilePath = "../ml_data/*.csv"

"""
   For testing purpose, we can choose to load a partial set of data
"""
LoadPartial=1
NumOfFileToLoad = 50
LogFileName = "../log_test.txt"
ResultFileName = "../result_file.txt"
UpdateAllLogsToFile = 1
WeightFileName = "../weightFile.txt"

"""
   Choose to train networks and then evolve them
   Or to train a single generation of neurons
"""
involveMutation = False

"""
   When we choose the mutation based approach,
   then we can choose, if we shall retrain a network, which is already trained.
"""
reTrainExistingNetworks = True
'''
Keras params
'''
EpochCount = 150
isKegarVerbose = False

