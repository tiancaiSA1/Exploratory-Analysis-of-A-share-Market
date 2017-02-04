#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 23:32:48 2017

@author: yinboya
"""
from __future__ import division
import trainAndtest_SET as TT
import rankmodel as rmo
import numpy as np

class testGBRank:
    '''
    init as train,test set index of the whole data
    input the list of train,test index
    '''
    def __init__(self,trainInd,testInd,filepath):
        self.trainI = trainInd
        self.testI = testInd
        self.filepath = filepath
        self.correctRate = []
        self.DataFP = TT.readFile(self.filepath)
        #self.trainNum = len(trainInd) # number of train set
        #self.testNum = len(testInd) # number of test set
    
    def trainRank(self):
        self.rmodel = rmo.GBRank()
        X = self.DataFP.TDataSET(self.trainI[0])
        y = X[:,-1]
        X = X[:,:-1]
        self.rmodel.fit(X,y)
        self.trainI = self.trainI[1:]
        for Ind in self.trainI:
            X = self.DataFP.TDataSET(Ind)
            y = X[:,-1]
            X = X[:,:-1]
            self.rmodel.update_fit(X,y)
    
    def testRank(self):
        if len(self.testI) == 1:
            Ind = self.testI
            X = self.DataFP.TDataSET(Ind)
            y = X[:,-1]
            X = X[:,:-1]
            Xs = []
            for i in xrange(X.shape[0]):
                for j in xrange(X.shape[0]):
                    if y[i] > y[j]:
                        Xs.append(X[i])  # y[i] > y[i+1]
                        Xs.append(X[j])
            l = np.ones(len(Xs), bool)
            Xs = np.asarray(Xs)  # as matrix
            H = self.rmodel.predict_proba(Xs).T  # as matrix
            for i in xrange(len(H)-2, -1, -2):
                assert (i % 2) == 0
                if H[i] >= H[i+1]:
                    l[i] = False
                    l[i+1] = False
            self.correctRate.append(1 - np.sum(l) / float(len(l)))
            return 0
        
        for Ind in self.testI:
            X = self.DataFP.TDataSET(Ind)
            y = X[:,-1]
            X = X[:,:-1]
            Xs = []
            for i in xrange(X.shape[0]):
                for j in xrange(X.shape[0]):
                    if y[i] > y[j]:
                        Xs.append(X[i])  # y[i] > y[i+1]
                        Xs.append(X[j])
            l = np.ones(len(Xs), bool)
            Xs = np.asarray(Xs)  # as matrix
            H = self.rmodel.predict_proba(Xs).T  # as matrix
            for i in xrange(len(H)-2, -1, -2):
                assert (i % 2) == 0
                if H[i] >= H[i+1]:
                    l[i] = False
                    l[i+1] = False
            self.correctRate.append(1 - np.sum(l) / float(len(l)))
      
    def clearlc(self):
        self.DataFP.clearlc()
        
    def test_Correct_Rate(self):
        return self.correctRate

    def hk(self, X):
        return(self.rmodel.predict_proba(X))
    
    def sortHK(self, X):
        hk_value = self.hk(X)
        return(np.argsort(-hk_value))
    
    