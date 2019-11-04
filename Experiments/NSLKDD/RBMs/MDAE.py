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

allFeatures = allData[:,0:122]

basicFeatures = allData[:,0:90]

contentFeatures = allData[:,90:103]

trafficFeatures = allData[:,103:122]

#-----读取权重参数函数-----
bwPath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/basicRBMPara/90_60_w.csv'
bb1Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/basicRBMPara/90_60_b1.csv'
bb2Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/basicRBMPara/90_60_b2.csv'

cwPath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/contentRBMPara/13_10_w.csv'
cb1Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/contentRBMPara/13_10_b1.csv'
cb2Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/contentRBMPara/13_10_b2.csv'

twPath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/trafficRBMPara/19_15_w.csv'
tb1Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/trafficRBMPara/19_15_b1.csv'
tb2Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/trafficRBMPara/19_15_b2.csv'

#-----保存权重参数函数-----
w2savePath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/multiDBNPara/95_20_w.csv'
b2savePath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/multiDBNPara/95_20_b1.csv'
b3savePath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/multiDBNPara/95_20_b2.csv'


#wsavePath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/noModalityRBMPara/122_80_w.csv'
#b1savePath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/noModalityRBMPara/122_80_b1.csv'
#b2savePath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/noModalityRBMPara/122_80_b2.csv'
#def writeCsv(savePath,data):
#    file  = open(savePath, 'w', newline='')
#    csvWriter = csv.writer(file)
#    count = 0
#    for row in data:
#        temp_row=np.array(row)
#        csvWriter.writerow(temp_row)
#        count +=1
#    file.close()

bw = np.loadtxt(bwPath, dtype=np.float32, delimiter=",", skiprows=0)
bb = np.loadtxt(bb1Path, dtype=np.float32, delimiter=",", skiprows=0)

cw = np.loadtxt(cwPath, dtype=np.float32, delimiter=",", skiprows=0)
cb = np.loadtxt(cb1Path, dtype=np.float32, delimiter=",", skiprows=0)

tw = np.loadtxt(twPath, dtype=np.float32, delimiter=",", skiprows=0)
tb = np.loadtxt(tb1Path, dtype=np.float32, delimiter=",", skiprows=0)

basic_x =tf.placeholder(dtype = tf.float32,shape=[None,90],name='basic_x')
content_x =tf.placeholder(dtype = tf.float32,shape=[None,13],name='content_x')
traffic_x =tf.placeholder(dtype = tf.float32,shape=[None,19],name='traffic_x')

bRBMHidden = tf.nn.sigmoid(tf.add(tf.matmul(basic_x,bw),bb))
binarybRBMSample = tf.nn.relu(tf.sign(bRBMHidden - tf.random_uniform(tf.shape(bRBMHidden))))

cRBMHidden = tf.nn.sigmoid(tf.add(tf.matmul(content_x,cw),cb))
cinarybRBMSample = tf.nn.relu(tf.sign(cRBMHidden - tf.random_uniform(tf.shape(cRBMHidden))))

tRBMHidden = tf.nn.sigmoid(tf.add(tf.matmul(traffic_x,tw),tb))
tinarybRBMSample = tf.nn.relu(tf.sign(tRBMHidden - tf.random_uniform(tf.shape(tRBMHidden))))

RBMSample = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    f1,f2,f3 = sess.run([binarybRBMSample,cinarybRBMSample,tinarybRBMSample], feed_dict={basic_x: basicFeatures,content_x: contentFeatures,traffic_x:trafficFeatures})
    RBMSample = f1
    RBMSample = np.hstack((RBMSample,f2))
    RBMSample = np.hstack((RBMSample,f3))
    

BBRBM = rbm(95,20,learning_rate=0.001,momentum=0.8,rbm_type='bbrbm',relu_hidden = True,init_method= 'uniform')
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


