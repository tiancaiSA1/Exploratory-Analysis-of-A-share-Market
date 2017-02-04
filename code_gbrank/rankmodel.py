#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 23:55:30 2017

@author: yinboya
"""
from __future__ import division
'''
 gbrank
'''
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import clone
#from utils import choose_threshold
#import itertools
import numpy as np



class GBRank(BaseEstimator, ClassifierMixin):
    """ rank
    # Start with an initial guess h0, for k = 1, 2, ...
     1) using h_{k-1} as the current approximation of h, we separate S into two
    disjoint sets, S+ = {(x_i, y_i) e S | h_{k-1}(x_i) >= h_{k-1}(y_i)+tau} and
    S- = {(x_i, y_i) e S | h_{k-1}(x_i) < h_{k-1}(y_i)+tau}
     2) fitting a regression function g_k(x) using GBT and the following training
    data {(x_i,h_{k-1}(yi)+tau), (y_i,h_{k-1}(x_i)-tau) | (xi, yi) e S-}
     3) forming (with normalization of the range of h_k) h_k(x) =
    k h_{k-1}(x)+eta g_k(x), k+1 where eta is a shrinkage factor

        Returns
        -------
        self
        """
    def __init__(self, base_estimator=None, max_its= 100, eta = 0.05, \
                 s_rate = 0.9):
        super(GBRank, self).__init__()
        self.g = [] # the set of classifierMixin
        self.eta = eta # 0.05
        self.kmax = max_its # maximum iter nums
        self.s_rate = s_rate 
        # iteration will stop when accuracy rate exceed the s_rate
        self.train_s_rate = []
        if base_estimator is None:
            base_estimator = GradientBoostingRegressor(learning_rate=0.05,
                                                       max_depth=7,
                                                       subsample=0.7,
                                                       min_samples_split=5)
        self.estimator = base_estimator

    # Estranhamente, este código até parece mais rápido que o anterior!

    def fit(self, X, y, tau = 1):
        #tau = 1  # [0,1]
        Xs = []
        H = []
        for i in xrange(len(X)):
            for j in xrange(len(X)):
                if y[i] > y[j]:
                    Xs.append(X[i])  # y[i] > y[i+1]
                    Xs.append(X[j])
                    H.append(2)
                    H.append(0)
        l = np.ones(len(H), bool)
        Xs = np.asarray(Xs)  # as matrix
        H = np.asarray(H).T  # as matrix
        NXs = Xs.shape[0]
        while self.kmax > len(self.g):
            g = clone(self.estimator).fit(Xs[l], H[l])
            self.g.append(g)
            l = np.ones(len(H), bool)
            H = self.predict_proba(Xs).T  # as matrix
            
            for i in xrange(len(H)-2, -1, -2):
                assert (i % 2) == 0
                if H[i] >= H[i+1]+tau:
                    l[i] = False
                    l[i+1] = False
                else:
                    Hi = H[i+1]+tau
                    H[i+1] = H[i]-tau
                    H[i] = Hi
            #if np.sum(l) == 0:  # converged
            #    break
            
            if np.sum(l) <= NXs * (1-self.s_rate):
                break
        self.train_s_rate.append(1 - np.sum(l) / NXs)
        #H = self.predict_proba(X)
        #self.th = choose_threshold(H, y)
        return self

    def update_fit(self, X, y, tau = 1):
        Xs = []
        for i in xrange(len(X)):
            for j in xrange(len(X)):
                if y[i] > y[j]:
                    Xs.append(X[i])  # y[i] > y[i+1]
                    Xs.append(X[j])
        l = np.ones(len(Xs), bool)
        Xs = np.asarray(Xs)  # as matrix
        NXs = Xs.shape[0]
        while self.kmax > len(self.g):
            H = self.predict_proba(Xs).T  # as matrix
            for i in xrange(len(H)-2, -1, -2):
                assert (i % 2) == 0
                if H[i] >= H[i+1]+tau:
                    l[i] = False
                    l[i+1] = False
                else:
                    Hi = H[i+1]+tau
                    H[i+1] = H[i]-tau
                    H[i] = Hi
            if np.sum(l) <= NXs * (1-self.s_rate):
                break
            
            g = clone(self.estimator).fit(Xs[l], H[l])
            self.g.append(g)
            l = np.ones(len(H), bool)
        self.train_s_rate.append(1 - np.sum(l) / NXs)
        return self
            
        

    def hk(self, X, k):
        if k == 0:
            s = self.g[0].predict(X)
            return s
        g = self.g[k].predict(X)
        h = self.hk(X, k-1)
        return (k*h + self.eta*g) / float(k+1)

    def predict_proba(self, X):
        # these are not probabilities, but I overloaded this function because
        # it makes it nicer to have a common interface for the ROC curves
        X = np.asarray(X)
        return self.hk(X, len(self.g)-1)

    def predict(self, X):
        return self.predict_proba(X)
        # return (self.predict_proba(X) >= self.th).astype(int)



