GlobalNNParamChoices = {
    'nb_neurons': [256, 512, 1024],
    'nb_layers': [1, 2, 3],
    'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
    'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                  'adadelta', 'adamax', 'nadam'],
}

GlobalWeightOfChoices = {
    'nb_neurons': [0.175, 0.425, 0.3],
    'nb_layers': [0.4, 0.3, 0.3],
    'activation': [0.4, 0.39, 0, 0.2],
    'optimizer': [0.14, 0.14, 0.14, 0.14,
                  0.14, 0.14, 0.14],
}
def RandomChoiceFn (key, paramsChoice):
    return random.choice(paramsChoice[key],size = 1,p=GlobalWeightOfChoices[key])


GlobalKnonwParams = [
   {'nb_layers': 1, 'activation': 'tanh', 'optimizer': 'adamax', 'nb_neurons': 512},
   {'nb_layers': 2, 'activation': 'relu', 'optimizer': 'nadam', 'nb_neurons': 512}
]

# Number of times to evolve the population.
Generations = 3
# Number of networks in population
Population = 20 + len(GlobalKnonwParams)

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
   If to removve Outliers
"""
RemoveOutlier = True

"""
   For testing purpose, we can choose to load a partial set of data
"""
LoadPartial=1
NumOfFileToLoad = 500
LogFileName = "../log_test.txt"
ResultFileName = "../result_file.txt"
UpdateAllLogsToFile = 1
WeightFileName = "../weightFile.txt"

"""
   Choose to train networks and then evolve them
   Or to train a single generation of neurons
"""
involveMutation = True

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

