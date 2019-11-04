# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 21:24:10 2019

@author: dabing
"""
import numpy as np
import csv

import loadData
from RBM import rbm
#--------------准备---------------
#-----读取数据-----
allData = loadData.allData

allFeatures = allData[:,0:122]

basicFeatures = allData[:,0:90]

contentFeatures = allData[:,90:103]

trafficFeatures = allData[:,103:122]

#-----保存权重参数函数-----
#wsavePath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/basicRBMPara/90_60_w.csv'
#b1savePath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/basicRBMPara/90_60_b1.csv'
#b2savePath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/basicRBMPara/90_60_b2.csv'

#wsavePath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/contentRBMPara/13_10_w.csv'
#b1savePath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/contentRBMPara/13_10_b1.csv'
#b2savePath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/contentRBMPara/13_10_b2.csv'

#wsavePath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/trafficRBMPara/19_15_w.csv'
#b1savePath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/trafficRBMPara/19_15_b1.csv'
#b2savePath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/trafficRBMPara/19_15_b2.csv'

wsavePath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/noModalityRBMPara/122_60_w.csv'
b1savePath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/noModalityRBMPara/122_60_b1.csv'
b2savePath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/noModalityRBMPara/122_60_b2.csv'
#def writeCsv(savePath,data):
#    file  = open(savePath, 'w', newline='')
#    csvWriter = csv.writer(file)
#    count = 0
#    for row in data:
#        temp_row=np.array(row)
#        csvWriter.writerow(temp_row)
#        count +=1
#    file.close()

no_GBRBM = rbm(122,60,learning_rate=0.001,momentum=0.8,rbm_type='gbrbm',relu_hidden = True)
no_GBRBM.plot=True
w, b1, b2= no_GBRBM.pretrain(allFeatures,batch_size=100,n_epoches=100)


#basic_GBRBM = rbm(90,70,learning_rate=0.001,momentum=0.8,rbm_type='gbrbm',relu_hidden = True)
#basic_GBRBM.plot=True
#w, b1, b2= basic_GBRBM.pretrain(basicFeatures,batch_size=100,n_epoches=100)

#content_GBRBM = rbm(13,10,learning_rate=0.001,momentum=0.8,rbm_type='gbrbm',relu_hidden = True)
#content_GBRBM.plot=True
#w, b1, b2= content_GBRBM.pretrain(contentFeatures,batch_size=100,n_epoches=100)

#traffic_GBRBM = rbm(19,15,learning_rate=0.001,momentum=0.8,rbm_type='gbrbm',relu_hidden = True)
#traffic_GBRBM.plot=True
#w, b1, b2 = traffic_GBRBM.pretrain(trafficFeatures,batch_size=100,n_epoches=100)

np.savetxt(wsavePath, w, delimiter=",")
np.savetxt(b1savePath, b1, delimiter=",")
np.savetxt(b2savePath, b2, delimiter=",")


