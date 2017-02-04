#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 02:57:23 2017

@author: yinboya
"""

'''
main
'''

import testRank as tR
import numpy as np

trainInd = np.array(range(2)) + 2
                   
testInd = np.array(range(5)) + 3

filepath = '/Users/yinboya/STUDY/QuantitativeInvestment/practice/outdata/'
testGBRank = tR.testGBRank(trainInd,testInd,filepath)
testGBRank.trainRank()
testGBRank.testRank()
testGBRank.clearlc()
testGBRank.test_Correct_Rate()

testI = testInd

testGBRank.rmodel
testGBRank.trainI[0]


