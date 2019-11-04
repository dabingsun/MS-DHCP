# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 16:01:18 2019

@author: dabing
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 21:28:31 2019

@author: dabing
"""
import numpy as np

def loadCsv(loadPath):
    data = np.loadtxt(loadPath, delimiter=',', skiprows = 0)
    return data

trainPath_2 = "E:/workspace for python/Experiment/data/NSL-KDD/03onehotNormalizationData/2ClassKDDTrain+.csv"
testPath_2 = "E:/workspace for python/Experiment/data/NSL-KDD/03onehotNormalizationData/2ClassKDDTest+.csv"

trainPath_5 = "E:/workspace for python/Experiment/data/NSL-KDD/03onehotNormalizationData/5ClassKDDTrain+.csv"
testPath_5 = "E:/workspace for python/Experiment/data/NSL-KDD/03onehotNormalizationData/5ClassKDDTest+.csv"

trainData_2 = loadCsv(trainPath_2)
testData_2 = loadCsv(testPath_2)

trainData_5 = loadCsv(trainPath_5)
testData_5 = loadCsv(testPath_5)

trainFeatures = trainData_2[:,0:122]
testFeatures = testData_2[:,0:122]

trainLabels_2 = trainData_2[:,122:]
testLabels_2 = testData_2[:,122:]

trainLabels_5 = trainData_5[:,122:]
testLabels_5 = testData_5[:,122:]