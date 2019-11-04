# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 21:28:32 2019

@author: dabing
"""

import numpy as np

def loadCsv(loadPath):
    data = np.loadtxt(loadPath, delimiter=',', skiprows = 0)
    return data

cicids2017Path = "E:/workspace for python/Experiment/data/CICIDS2017/03standData/2ClassTrain.csv"

allData = loadCsv(cicids2017Path)