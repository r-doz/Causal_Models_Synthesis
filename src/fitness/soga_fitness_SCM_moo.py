from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
#from fitness.base_ff_classes.moo_ff import moo_ff
import time as timeit
import signal
import sys
import numpy as np
import re
import torch
from fitness.soga_fitness_SCM import soga_fitness_SCM
from fitness.minimise_causal_arrows import minimise_causal_arrows

torch.set_default_dtype(torch.float64)
# Define a custom exception for timeouts
class TimeoutException(Exception):
    pass

def timeout_handler():
    raise TimeoutException()


class soga_fitness_SCM_moo(base_ff):

    multi_objective = True

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        self.num_obj = 2
        fitness1 = soga_fitness_SCM()
        fitness2 = minimise_causal_arrows()

        self.fitness_functions = [fitness1, fitness2]
        self.default_fitness = [torch.tensor(-1e6), float('nan')]

    def evaluate(self, ind, **kwargs):
        p = ind.phenotype
        fitness2 = -minimise_causal_arrows().evaluate(ind, **kwargs)
        try:
            fitness1 = soga_fitness_SCM().evaluate(ind, **kwargs)
            fitness = [fitness1, fitness2]         
        except TimeoutException as e:
            #print("Caught TimeoutException")
            fitness = [torch.tensor(-1e6), fitness2]
        except:
            #print("Caught general SOGA exception")
            fitness = [torch.tensor(-1e6), fitness2]
            #I do not define the indiviaduals as invalid in order to allow crossover
            #if not hasattr(params['FITNESS_FUNCTION'], "multi_objective"):
                #stats['invalids'] += 1
        #finally:
            #timer.cancel()
        
    
        return fitness
    
    @staticmethod
    def value(fitness_vector, objective_index):
        """
        This is a static method required by NSGA-II for sorting populations
        based on a given fitness function, or for returning a given index of a
        population based on a given fitness function.
        
        :param fitness_vector: A vector/list of fitnesses.
        :param objective_index: The index of the desired fitness.
        :return: The fitness at the objective index of the fitness vector.
        """

        if not isinstance(fitness_vector, list):
            return float("inf")

        return fitness_vector[objective_index]