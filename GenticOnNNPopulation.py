"""
Placeholder for the main gentic algorithm that eveloves the entire population
"""
import logging
from NNPorpulation import NeuralNetwork
from NNPorpulation import NNDb
from Params        import GlobalNNParamChoices as nnParams
from Params        import RandomChoiceFn as randomFn
from Params        import FitnessPopulationPercent as parentPercent
from Params        import Population     as populationCount
from Params        import FitnessPopulationPercent as fitnessPercent

import random

"""
  @brief Breed a new child from 2 parents.
         This include choose following parameters from each
           activation, num of layer, num of neurons , optimizer
         This child will be fed to training later
"""
def genrateChildParamFromParent(parent1, parent2):
   childParam = {}
   for key in nnParams:
      print(parent1._network[key])
      param = random.choice([parent1._network[key], parent2._network[key]])
      #childParam.append(param)
      childParam[key] = param
   return childParam

"""
   @brief Update few child neural networks to a pupulation of fit and trained
          neural networks
   @param[in]: fittestSet: set of fit parent, child will be appended to this population
   @param[in]  maxNumChild: how much can this population fit
"""
def BreedChild(fittestSet, maxNumChild):
   parentLen = len(fittestSet)
   print("Breeding Childs"+" parent set len:"+str(parentLen)+" max child:"+str(maxNumChild))
   idx = parentLen
   while (idx < maxNumChild):
      idx += 1
      parentNN1 = random.randint(0, parentLen-1)
      parentNN2 = random.randint(0, parentLen-1)
      print("Mutating:"+" Parent:"+str(parentNN1)+"with Parent:"+str(parentNN2))

      if (parentNN1 != parentNN2):
         newChildParam = genrateChildParamFromParent(fittestSet[parentNN1],
                                                     fittestSet[parentNN2])
         """
         generate neuralNetwork with this param
         """
         childNN = NeuralNetwork(newChildParam, None)
         print("Created Child")
         childNN.describe()
         fittestSet.append(childNN)



"""
    @brief Mutate a particular NN
           This is kostly applied to one child NN
           choose of params randomly like activation, optimizer, num of layers and change that.
"""
def MutateNN(nn):
   print("Before mutation")
   nn.describe()
   # Choose a random key.
   mutation = random.choice(list(nn._nn_param_choices.keys()))
   # Mutate one of the params.
   mutatedParam = random.choice(nn._nn_param_choices[mutation])
   nn.updateParams(mutation, mutatedParam)
   print("After Mutation")
   nn.describe()

"""
   @class This is the class to host population of a generation of NN
"""
class Generation:
   _population = []
   _fitnessFn = None
   _fitnessLevel = None
   def __init__(self, population, fitnessFn):
      if population is not None:
         self._population = population
      self._fitnessFn = fitnessFn

   def fitness(self):
      fitnessLevel = 0
      count = 0
      for nn in self._population:
         fitnessLevel = fitnessLevel + self._fitnessFn(nn)
         count = count + 1
      self._fitnessLevel = fitnessLevel / count
      print("totalFitness:"+str(fitnessLevel)+"count:"+str(count))
      return self._fitnessLevel

   def evolve(self, breedFn=None, mutationFn=None):
      """
      Sort based on accuracy the first  5% of the population
      """
      numNNToChoose = int((float(populationCount) * fitnessPercent) / 100)

      maxChildPercent = fitnessPercent + 10
      if (maxChildPercent > 100):
         maxChildPercent = 90

      maxChild = int((float(populationCount) * maxChildPercent) / 100)
      print("Evolving population len:" + str(len(self._population)))
      print("Choose top:" + str(numNNToChoose) + " as parent")
      print("MaxChild:" + str(maxChild))

      print("Evolving Db")
      gradedNN = [(self._fitnessFn(nn), nn) for nn in self._population]

      gradedNN = [x[1] for x in sorted(gradedNN, key=lambda x: x[0], reverse=True)]
      print("Sorted on Accuracy")
      print(gradedNN)

      fittestNNPopulation = gradedNN[:numNNToChoose]
      fittestNNPopulation[1].describe()
      print("Describing top NN")
      for idx in range(0, len(fittestNNPopulation) - 1):
         fittestNNPopulation[idx].describe()

      """
      Add a few of the least fittest to the list
      """
      """
      for nn in gradedNN[numNNToChoose:]:
         if 5 > random.random():
            fittestNNPopulation.append(nn)
      """
      if breedFn is not None:
         breedFn(fittestNNPopulation, maxChild)
      else:
         BreedChild(fittestNNPopulation, maxChild)

      """
         Mutate a few child
      """
      return fittestNNPopulation

   def train(self,trainFn):
      for nn in self._population:
         trainFn(nn)
