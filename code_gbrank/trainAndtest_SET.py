#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 20:24:54 2017

@author: yinboya
"""

import os
import linecache as lc
import numpy as np

class readFile:
    def __init__(self ,filepath):
        self.filepath = filepath
        self.file = []
    
    def eachFile(self):
        self.pathDir = os.listdir(self.filepath)
        
    def TDataSET(self,i):
        TSET = []
        self.eachFile()
        for filename in self.pathDir:
            #print list(eval(lc.getline(filename,i)))
            try:
                TSET.append(list(eval(lc.getline(self.filepath + filename,i))))
            except Exception:
                continue
        return(np.array(TSET))
    
    def clearlc(self):
        lc.clearcache()
