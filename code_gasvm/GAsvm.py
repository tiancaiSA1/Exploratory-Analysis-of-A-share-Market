#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 00:02:58 2017

@author: yinboya
"""
# Read Data
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

filepath = '/Users/yinboya/STUDY/QuantitativeInvestment/practice/outdata/'
sym = pd.read_csv(filepath + 'sym_1.csv',iterator=True)
trainNum = 1000
testNum = 500
sym = sym.get_chunk(trainNum + testNum)
sym = np.array(sym)
trainX, trainY = sym[range(trainNum),:-1],sym[range(trainNum),-1]*100
testX, testY = sym[np.array(range(testNum)) + trainNum,:-1],\
                   sym[np.array(range(testNum)) + trainNum,-1]*100

# Setting Things Up
import random
from deap import base
from deap import creator
from deap import tools

# Creator
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Attribute generator 
def svm_C(N = 10000):
    return(random.random() * N)

def svm_Kernel():
    i = random.randint(0,3)
    return(i)

toolbox.register("attr_svm_C", svm_C, 100)
toolbox.register("attr_svm_Kernel", svm_Kernel)

# Structure initializers
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_svm_C, toolbox.attr_svm_Kernel), 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# The Evaluation Function
from sklearn import svm
def svmc_evaluation(individual):
    N_SPLITS = 2
    kf = KFold(n_splits = N_SPLITS)
    cv_mseValue = 0
    Kernel = ['linear','rbf','poly','sigmoid']
    fc = svm.SVR(C = individual[0], kernel = Kernel[individual[1]])
    for train, test in kf.split(trainX):
        fc.fit(trainX[train,:], trainY[train])
        cv_mseValue += sum((trainY[test] - fc.predict(trainX[test,:])) ** 2) / len(test)
    cv_mseValue = cv_mseValue / N_SPLITS
    return(cv_mseValue,)

# The mutate Function
def svmc_mutation(individual):
    delta = 10 # the delta of Guassian
    if random.random() < 0.5:
        tmp = random.gauss(individual[0],delta)
        while tmp < 0:
            tmp = random.gauss(individual[0],delta)
        individual[0] = tmp
    else:
        individual[1] = random.randint(0,3)

# The selection

# The Genetic Operators
toolbox.register("evaluate", svmc_evaluation)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", svmc_mutation)
toolbox.register("select", tools.selTournament, tournsize = 3)


def main(NGEN = 10,CXPB = 0.5, MUTPB = 0.2):
    # Creating the Population
    pop = toolbox.population(n=5)
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    # Begin the evolution
    for g in range(NGEN):
        print("-- Generation %i --" % g)
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        print("  Parameter %s" % offspring)

if __name__ == '__main__':
    main()