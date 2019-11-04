# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 13:29:28 2019

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
#trainbasicFeatures = trainData_2[:,0:169]
#traincontentFeatures = trainData_2[:,169:184]
#traintrafficFeatures = trainData_2[:,184:196]
#
#testbasicFeatures = testData_2[:,0:169]
#testcontentFeatures = testData_2[:,169:184]
#testtrafficFeatures = testData_2[:,184:196]
#
#trainLabels_2 = trainData_2[:,196:]
#testLabels_2 = testData_2[:,196:]

trainData_10 = loadCsv(trainPath_10)
testData_10 = loadCsv(testPath_10)

trainbasicFeatures = trainData_10[:,0:169]
traincontentFeatures = trainData_10[:,169:184]
traintrafficFeatures = trainData_10[:,184:196]

testbasicFeatures = testData_10[:,0:169]
testcontentFeatures = testData_10[:,169:184]
testtrafficFeatures = testData_10[:,184:196]

trainLabels_10 = trainData_10[:,196:]
testLabels_10 = testData_10[:,196:]