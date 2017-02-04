#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 03:28:04 2017

@author: yinboya
"""


from __future__ import division
import numpy as np
from sklearn.model_selection import KFold
import GenDataSet

I = 1
'''
represent the(classifcation of regression) model:
    1: classfication
    2: regression
'''
popNum = 10 # Population
N_splits = 2 # Crossvalidation fold number
iter_NGEN = 10 # Generation number
iter_CXPB = 0.8 # Crossover probability
iter_MUTPB = 0.9 # Mutation probability 

classMaxThreshold = 0.001
classMinThreshold = -0.001

# Read Data
if I == 1:
    filepath = '/Users/yinboya/STUDY/QuantitativeInvestment/practice/outdata/'
    name = 'sym_1.csv'
    myrf = GenDataSet.csv2tset(filepath = filepath, name = name)
    trainNum = 1000
    testNum = 500
    trainX, trainY = myrf.tData_Classification(trainNum,
                                               Class_MaxThreshold = classMaxThreshold,
                                               Class_MinThreshold = classMinThreshold)
    testX, testY = myrf.tData_Classification(testNum,
                                             Class_MaxThreshold = classMaxThreshold,
                                             Class_MinThreshold = classMinThreshold)
else:
    filepath = '/Users/yinboya/STUDY/QuantitativeInvestment/practice/outdata/'
    name = 'sym_1.csv'
    myrf = GenDataSet.csv2tset(filepath = filepath, name = name)
    trainNum = 1000
    testNum = 500
    trainX, trainY = myrf.tData_Regression(trainNum)
    testX, testY = myrf.tData_Regression(testNum)

# Setting Things Up
import random
from deap import base
from deap import creator
from deap import tools

# Creator
if I == 1:
    creator.create("FitnessMulti", base.Fitness, weights=(1.0,1.0,1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
else:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

'''
 Attribute generator 
'''
"""
General Parameters:
    booster[default = gbtree]
    silent[default = 0]
    nthread[default to maximum number of threads available if not set]
"""

"""
Booster Parameters:
    eta[default = 0.3]
    min_child_weight[default = 1]:
        define the minimum sum of weight of all observations required in a child
        This is similar to min_child_leaf in GBM but not exactly. This refers to 
        min “sum of weights” of observations while GBM has min “number of observations”.
        Used to control over-fitting. Higher values prevent a model from learning 
        relations which might be highly specific to the particular sample selected 
        for a tree.
        Too high values can lead to under-fitting hence, it should be tuned using CV.
    max_depth[default = 6]:
        typical:3-10
    max_leaf_nodes:
        If this is defined, GBM will ignore max_depth.
    gamma[default = 0]:
        A node is split only when the resulting split gives a positive reduction in 
        the loss function. Gamma specifies the minimum loss reduction required to 
        make a split.
        Makes the algorithm conservative. The values can vary depending on the loss
        function and should be tuned.
    #max_delta_step[default=0]:
        In maximum delta step we allow each tree’s weight estimation to be.
        This is generally not used but you can explore further if you wish.
    subsample [default=1]:
        Same as the subsample of GBM. Denotes the fraction of observations 
        to be randomly samples for each tree.
        Typical values: 0.5-1
    colsample_bytree[default=1]:
        Similar to max_features in GBM. Denotes the fraction of columns to be 
        randomly samples for each tree.
        Typical values: 0.5-1
    #colsample_bylevel [default=1]
    #lambda [default=1]:
        L2 regularization term on weights (analogous to Ridge regression)
        This used to handle the regularization part of XGBoost. Though many 
        data scientists don’t use it often, it should be explored to reduce overfitting.
    #alpha [default=0]:
        L1 regularization term on weight (analogous to Lasso regression)
        Can be used in case of very high dimensionality so that the algorithm runs 
        faster when implemented
    #scale_pos_weight [default=1]:
        A value greater than 0 should be used in case of high class imbalance as it 
        helps in faster convergence.
"""
def xgb_eta(MAXeta = 0.2, MINeta = 0.01):
    return((MAXeta - MINeta)*random.random() + MINeta)

def xgb_MinChildWeight(MAXmcw = 20, MINmcw = 5):
    return(random.randint(MINmcw,MAXmcw))

def xgb_maxdepth(MAXmd = 13, MINmd = 3):
    return(random.randint(MINmd,MAXmd))

def xgb_subsample(MAXss = 1, MINss = 0.4):
    return((MAXss - MINss)*random.random() + MINss)

def xgb_colsample(MAXcs = 1, MINcs = 0.4):
    return((MAXcs - MINcs)*random.random() + MINcs)

# register xgb parameters
toolbox.register("attr_xgb_eta", xgb_eta)
toolbox.register("attr_xgb_MinChildWeight", xgb_MinChildWeight)
toolbox.register("attr_xgb_maxdepth", xgb_maxdepth)
toolbox.register("attr_xgb_subsample", xgb_subsample)
toolbox.register("attr_xgb_colsample", xgb_colsample)


"""
Learning Task Parameters:
    #objective [default=reg:linear]:
        This defines the loss function to be minimized. Mostly used values are:
            binary:logistic –logistic regression for binary classification, returns p
            redicted probability (not class)
            
            multi:softmax –multiclass classification using the softmax objective, 
            returns predicted class (not probabilities):
                you also need to set an additional num_class (number of classes) 
                parameter defining the number of unique classes
            
            multi:softprob –same as softmax, but returns predicted probability 
            of each data point belonging to each class.
    #eval_metric [ default according to objective ]:
        The metric to be used for validation data.
        The default values are rmse for regression and error for classification.
        Typical values are:
            rmse – root mean square error
            mae – mean absolute error
            logloss – negative log-likelihood
            error – Binary classification error rate (0.5 threshold)
            merror – Multiclass classification error rate
            mlogloss – Multiclass logloss
            auc: Area under the curve
    #seed [default=0]:
"""
"""
If you’ve been using Scikit-Learn till now, 
these parameter names might not look familiar. 
A good news is that xgboost module in python has an 
sklearn wrapper called XGBClassifier. It uses sklearn 
style naming convention. The parameters names which will change are:

eta –> learning_rate
lambda –> reg_lambda
alpha –> reg_alpha
"""


# Structure initializers
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_xgb_eta,toolbox.attr_xgb_MinChildWeight,
                  toolbox.attr_xgb_maxdepth,toolbox.attr_xgb_subsample,
                  toolbox.attr_xgb_colsample), 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# The Evaluation Function
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor

def XGB_class_evaluation(individual):
    N_SPLITS = N_splits
    kf = KFold(n_splits = N_SPLITS)
    fc = XGBClassifier(learning_rate = individual[0], n_estimators = 100,
                       silent = True,
                       nthread = -1, gamma = 0,
                       min_child_weight = individual[1],max_depth = individual[2],
                       subsample = individual[3],colsample_bylevel = individual[4],
                       seed = 0)
    M_pos = 0
    M_mid = 0
    M_neg = 0
    for train, test in kf.split(trainX):
        fc.fit(trainX[train,:], trainY[train])
        testY_pre = fc.predict(trainX[test,:])
        Ind_pos = (trainY[test] == 1)
        Ind_mid = (trainY[test] == 0)
        Ind_neg = (trainY[test] == -1)
        M_pos += len(np.where(np.array(testY_pre[Ind_pos]) == 1)[0]) / len(np.where(Ind_pos)[0])
        M_mid += len(np.where(np.array(testY_pre[Ind_mid]) == 0)[0]) / len(np.where(Ind_mid)[0])
        M_neg += len(np.where(np.array(testY_pre[Ind_neg]) == -1)[0]) / len(np.where(Ind_neg)[0])
        
    correct = map(lambda x : x / N_SPLITS, [M_pos, M_mid, M_neg])
    return(tuple(correct))

def XGB_reg_evaluation(individual):
    N_SPLITS = N_splits
    kf = KFold(n_splits = N_SPLITS)
    cv_mseValue = 0
    fc = XGBRegressor(learning_rate = individual[0], n_estimators = 100,
                       silent = True, objective = "reg:linear",
                       nthread = -1, gamma = 0,
                       min_child_weight = individual[1],max_depth = individual[2],
                       subsample = individual[3],colsample_bylevel = individual[4],
                       seed = 0)
    for train, test in kf.split(trainX):
        fc.fit(trainX[train,:], trainY[train])
        cv_mseValue += sum((trainY[test] - fc.predict(trainX[test,:])) ** 2) / len(test)
    cv_mseValue = cv_mseValue / N_SPLITS
    return(cv_mseValue,)

# The mutate Function
def xgb_mutation(individual):
    N = len(individual)
    rn = random.randint(1,N)
    if rn == 1 :
        individual[0] = toolbox.attr_xgb_eta()
    elif rn == 2 :
        individual[1] = toolbox.attr_xgb_MinChildWeight()
    elif rn == 3 :
        individual[2] = toolbox.attr_xgb_maxdepth()
    elif rn == 4 :
        individual[3] = toolbox.attr_xgb_subsample()
    elif rn == 5 :
        individual[4] = toolbox.attr_xgb_colsample()


# The Genetic Operators
if I == 1:
    toolbox.register("evaluate", XGB_class_evaluation)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", xgb_mutation)
    toolbox.register("select", tools.selTournament, tournsize = 3)
    def main(NGEN = iter_NGEN, CXPB = iter_CXPB, MUTPB = iter_MUTPB):
        # Creating the Population
        pop = toolbox.population(n=popNum)
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
            fits = [ind.fitness.values for ind in pop]
            fits = np.array(fits)
            
            mean = np.mean(fits, axis = 0)
            std = np.std(fit, axis = 0)
            
            print("  Min %s" % np.min(fits,axis = 0))
            print("  Max %s" % np.max(fits,axis = 0))
            print("  Avg %s" % mean)
            print("  Std %s" % std)
            print("  Parameter %s" % tools.selBest(offspring,1))
            
else:
    toolbox.register("evaluate", XGB_reg_evaluation)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", xgb_mutation)
    toolbox.register("select", tools.selTournament, tournsize = 3)
    def main(NGEN = iter_NGEN, CXPB = iter_CXPB, MUTPB = iter_MUTPB):
        # Creating the Population
        pop = toolbox.population(n=popNum)
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
            print("  Parameter %s" % offspring[np.argmin(fits)])



if __name__ == '__main__':
    main()
