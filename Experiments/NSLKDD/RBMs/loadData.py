# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 21:28:31 2019

@author: dabing
"""

import numpy as np

def loadCsv(loadPath):
    data = np.loadtxt(loadPath, delimiter=',', skiprows = 0)
    return data

nslkddPath = "E:/workspace for python/Experiment/data/NSL-KDD/03onehotNormalizationData/2ClassKDDTrain+.csv"

allData = loadCsv(nslkddPath)