# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 21:28:32 2019

@author: dabing
"""

import numpy as np

def loadCsv(loadPath):
    data = np.loadtxt(loadPath, delimiter=',', skiprows = 0)
    return data

nuswnb15Path = "E:/workspace for python/Experiment/data/UNSW-NB15/03onehotNormalizationData/2ClassTrain.csv"

allData = loadCsv(nuswnb15Path)