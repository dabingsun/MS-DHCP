# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 13:29:28 2019

@author: dabing
"""

import numpy as np

def loadCsv(loadPath):
    data = np.loadtxt(loadPath, delimiter=',', skiprows = 0)
    return data

trainPath_2 = "E:/workspace for python/Experiment/data/CICIDS2017/03standData/2ClassTrain.csv"
testPath_2 = "E:/workspace for python/Experiment/data/CICIDS2017/03standData/2ClassTest.csv"

trainPath_8 = "E:/workspace for python/Experiment/data/CICIDS2017/03standData/8ClassTrain.csv"
testPath_8 = "E:/workspace for python/Experiment/data/CICIDS2017/03standData/8ClassTest.csv"

#trainData_2 = loadCsv(trainPath_2)
#testData_2 = loadCsv(testPath_2)
#
#trainFeatures = trainData_2[:,0:77]
#
#testFeatures = testData_2[:,0:77]
#
#
#trainLabels_2 = trainData_2[:,77:]
#testLabels_2 = testData_2[:,77:]
#
trainData_8 = loadCsv(trainPath_8)
testData_8 = loadCsv(testPath_8)

trainFeatures = trainData_8[:,0:77]

testFeatures = testData_8[:,0:77]

trainLabels_8 = trainData_8[:,77:]
testLabels_8 = testData_8[:,77:]