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

trainPath_2 = "E:/workspace for python/Experiment/data/UNSW-NB15/03onehotNormalizationData/2ClassTrain.csv"
testPath_2 = "E:/workspace for python/Experiment/data/UNSW-NB15/03onehotNormalizationData/2ClassTest.csv"

trainPath_10 = "E:/workspace for python/Experiment/data/UNSW-NB15/03onehotNormalizationData/10ClassTrain.csv"
testPath_10 = "E:/workspace for python/Experiment/data/UNSW-NB15/03onehotNormalizationData/10ClassTest.csv"

#trainData_2 = loadCsv(trainPath_2)
#testData_2 = loadCsv(testPath_2)
#
#trainFeatures = trainData_2[:,0:196]
#testFeatures = testData_2[:,0:196]
#
#trainLabels_2 = trainData_2[:,196:]
#testLabels_2 = testData_2[:,196:]

trainData_10 = loadCsv(trainPath_10)
testData_10 = loadCsv(testPath_10)

trainFeatures = trainData_10[:,0:196]
testFeatures = testData_10[:,0:196]

trainLabels_10 = trainData_10[:,196:]
testLabels_10 = testData_10[:,196:]