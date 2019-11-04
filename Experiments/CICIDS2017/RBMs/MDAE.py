# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 10:13:12 2019

@author: dabing
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 21:24:10 2019

@author: dabing
"""
import tensorflow as tf
import numpy as np
import csv

import loadData
from RBM import rbm
#--------------准备---------------
#-----读取数据-----
allData = loadData.allData

allFeatures = allData[:,0:77]

contentFeatures = allData[:,0:34]

trafficFeatures = allData[:,34:77]

#-----读取权重参数函数-----
cwPath = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/RBMs/contentRBMPara/34_30_w.csv'
cb1Path = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/RBMs/contentRBMPara/34_30_b1.csv'
cb2Path = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/RBMs/contentRBMPara/34_30_b2.csv'

twPath = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/RBMs/trafficRBMPara/43_40_w.csv'
tb1Path = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/RBMs/trafficRBMPara/43_40_b1.csv'
tb2Path = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/RBMs/trafficRBMPara/43_40_b2.csv'

#-----保存权重参数函数-----
w2savePath = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/RBMs/multiDBNPara/70_50_w.csv'
b2savePath = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/RBMs/multiDBNPara/70_50_b1.csv'
b3savePath = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/RBMs/multiDBNPara/70_50_b2.csv'


#wsavePath = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/RBMs/noModalityRBMPara/70_30_w.csv'
#b1savePath = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/RBMs/noModalityRBMPara/70_30_b1.csv'
#b2savePath = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/RBMs/noModalityRBMPara/70_30_b2.csv'
#def writeCsv(savePath,data):
#    file  = open(savePath, 'w', newline='')
#    csvWriter = csv.writer(file)
#    count = 0
#    for row in data:
#        temp_row=np.array(row)
#        csvWriter.writerow(temp_row)
#        count +=1
#    file.close()

cw = np.loadtxt(cwPath, dtype=np.float32, delimiter=",", skiprows=0)
cb = np.loadtxt(cb1Path, dtype=np.float32, delimiter=",", skiprows=0)

tw = np.loadtxt(twPath, dtype=np.float32, delimiter=",", skiprows=0)
tb = np.loadtxt(tb1Path, dtype=np.float32, delimiter=",", skiprows=0)

content_x =tf.placeholder(dtype = tf.float32,shape=[None,34],name='content_x')
traffic_x =tf.placeholder(dtype = tf.float32,shape=[None,43],name='traffic_x')

cRBMHidden = tf.nn.sigmoid(tf.add(tf.matmul(content_x,cw),cb))
cinarybRBMSample = tf.nn.relu(tf.sign(cRBMHidden - tf.random_uniform(tf.shape(cRBMHidden))))

tRBMHidden = tf.nn.sigmoid(tf.add(tf.matmul(traffic_x,tw),tb))
tinarybRBMSample = tf.nn.relu(tf.sign(tRBMHidden - tf.random_uniform(tf.shape(tRBMHidden))))

RBMSample = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    f2,f3 = sess.run([cinarybRBMSample,tinarybRBMSample], feed_dict={content_x: contentFeatures,traffic_x:trafficFeatures})
    RBMSample = f2
    RBMSample = np.hstack((RBMSample,f3))
    

BBRBM = rbm(70,50,learning_rate=0.001,momentum=0.8,rbm_type='bbrbm',relu_hidden = True,init_method= 'uniform')
BBRBM.plot=True
w2, b2, b3= BBRBM.pretrain(RBMSample,batch_size=100,n_epoches=100)


#no_GBRBM = rbm(122,80,learning_rate=0.001,momentum=0.8,rbm_type='gbrbm',relu_hidden = True)
#no_GBRBM.plot=True
#w, b1, b2= no_GBRBM.pretrain(allFeatures,batch_size=100,n_epoches=100)


#basic_GBRBM = rbm(90,70,learning_rate=0.001,momentum=0.8,rbm_type='gbrbm',relu_hidden = True)
#basic_GBRBM.plot=True
#w, b1, b2= basic_GBRBM.pretrain(basicFeatures,batch_size=100,n_epoches=100)

#content_GBRBM = rbm(13,10,learning_rate=0.001,momentum=0.8,rbm_type='gbrbm',relu_hidden = True)
#content_GBRBM.plot=True
#w, b1, b2= content_GBRBM.pretrain(contentFeatures,batch_size=100,n_epoches=100)

#traffic_GBRBM = rbm(19,15,learning_rate=0.001,momentum=0.8,rbm_type='gbrbm',relu_hidden = True)
#traffic_GBRBM.plot=True
#w, b1, b2 = traffic_GBRBM.pretrain(trafficFeatures,batch_size=100,n_epoches=100)

np.savetxt(w2savePath, w2, delimiter=",")
np.savetxt(b2savePath, b2, delimiter=",")
np.savetxt(b3savePath, b3, delimiter=",")


