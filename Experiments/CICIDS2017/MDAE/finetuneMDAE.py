# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 10:09:15 2019

@author: dabing
"""

import tensorflow as tf 
import numpy as np
from loadData import traincontentFeatures,traintrafficFeatures 

import matplotlib.pyplot as plt 


#-----读取权重参数路径-----
cwloadPath = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/RBMs/contentRBMPara/34_30_w.csv'
cb1loadPath = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/RBMs/contentRBMPara/34_30_b1.csv'
cb2loadPath = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/RBMs/contentRBMPara/34_30_b2.csv'

twloadPath = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/RBMs/trafficRBMPara/43_40_w.csv'
tb1loadPath = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/RBMs/trafficRBMPara/43_40_b1.csv'
tb2loadPath = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/RBMs/trafficRBMPara/43_40_b2.csv'

w2loadPath = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/RBMs/multiDBNPara/70_50_w.csv'
b2loadPath = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/RBMs/multiDBNPara/70_50_b1.csv'
b3loadPath = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/RBMs/multiDBNPara/70_50_b2.csv'

#-----保存权重参数路径-----
cw1Path = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/MDAE/fine_tune/70_50cw1.csv'
cb1Path = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/MDAE/fine_tune/70_50cb1.csv'

tw1Path = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/MDAE/fine_tune/70_50tw1.csv'
tb1Path = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/MDAE/fine_tune/70_50tb1.csv'

w2Path = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/MDAE/fine_tune/70_50w2.csv'
b2Path = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/MDAE/fine_tune/70_50b2.csv'


#-----读取数据-----
c_train,t_train = traincontentFeatures,traintrafficFeatures 

#-----参数初始化-----

#-----自定义参数初始化函数，标准的均匀均匀分布-----
def xavier_init(fan_in,fan_out,constant = 1):
    low = -constant * np.sqrt(6.0/(fan_in + fan_out))
    high = constant * np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype = tf.float32)
#-----读取预训练参数-----

#w1 = np.loadtxt(wsavePath, dtype=np.float32, delimiter=",", skiprows=0)
#b1 = np.loadtxt(b1savePath, dtype=np.float32, delimiter=",", skiprows=0)
#b2 = np.loadtxt(b2savePath, dtype=np.float32, delimiter=",", skiprows=0)
#w2 = np.transpose(w1)

#-----当需要共享参数的时候，用get_variable

cw1 = np.loadtxt(cwloadPath, dtype=np.float32, delimiter=",", skiprows=0)
cb1 = np.loadtxt(cb1loadPath, dtype=np.float32, delimiter=",", skiprows=0)
cb4 = np.loadtxt(cb2loadPath, dtype=np.float32, delimiter=",", skiprows=0)
#cw4 = np.transpose(cw1)

tw1 = np.loadtxt(twloadPath, dtype=np.float32, delimiter=",", skiprows=0)
tb1 = np.loadtxt(tb1loadPath, dtype=np.float32, delimiter=",", skiprows=0)
tb4 = np.loadtxt(tb2loadPath, dtype=np.float32, delimiter=",", skiprows=0)
#tw4 = np.transpose(tw1)

w2 = np.loadtxt(w2loadPath, dtype=np.float32, delimiter=",", skiprows=0)
b2 = np.loadtxt(b2loadPath, dtype=np.float32, delimiter=",", skiprows=0)
b3 = np.loadtxt(b3loadPath, dtype=np.float32, delimiter=",", skiprows=0)
#w3 = np.transpose(w2)

ae_cw1 = tf.Variable(cw1,name="ae_cw1")
ae_cb1 = tf.Variable(cb1,name = "ae_cb1")
#ae_cw4 = tf.Variable(cw4,name="ae_cw4")
ae_cb4 = tf.Variable(cb4,name = "ae_cb4")

ae_tw1 = tf.Variable(tw1,name="ae_tw1")
ae_tb1 = tf.Variable(tb1,name = "ae_tb1")
#ae_tw4 = tf.Variable(tw4,name="ae_tw4")
ae_tb4 = tf.Variable(tb4,name = "ae_tb4")

ae_w2 = tf.Variable(w2,name = "ae_w2")
ae_b2 =tf.Variable(b2,name="ae_b2")
#ae_w3 = tf.Variable(w3,name = "ae_w3")
ae_b3 =tf.Variable(b3,name="ae_b3")

content_x =tf.placeholder(dtype = tf.float32,shape=[None,34],name='content_x')
traffic_x =tf.placeholder(dtype = tf.float32,shape=[None,43],name='traffic_x')
#basic_x content_x traffic_x
def MDAE_inference(content_x,traffic_x):
    
    cHidden1 = tf.nn.sigmoid(tf.add(tf.matmul(content_x,ae_cw1),ae_cb1))
    tHidden1 = tf.nn.sigmoid(tf.add(tf.matmul(traffic_x,ae_tw1),ae_tb1))
    hidden1 = tf.concat([cHidden1,tHidden1],1)
    
    hidden2 = tf.nn.sigmoid(tf.add(tf.matmul(hidden1,ae_w2),ae_b2))
    
    hidden3 = tf.nn.sigmoid(tf.add(tf.matmul(hidden2,tf.transpose(ae_w2)),ae_b3))
#    hidden3 = tf.add(tf.matmul(hidden2,ae_w3),ae_b3)
    
    cHidden3 = tf.slice(hidden3,[0,0],[-1,30])
    tHidden3 = tf.slice(hidden3,[0,30],[-1,-1])
    

    cReconstruction = tf.nn.sigmoid(tf.add(tf.matmul(cHidden3,tf.transpose(ae_cw1)),ae_cb4))
    tReconstruction = tf.nn.sigmoid(tf.add(tf.matmul(tHidden3,tf.transpose(ae_tw1)),ae_tb4))
#    bReconstruction = tf.add(tf.matmul(bHidden3,ae_bw4,ae_bb4))
#    cReconstruction = tf.add(tf.matmul(cHidden3,ae_cw4,ae_cb4))
#    tReconstruction = tf.add(tf.matmul(tHidden3,ae_tw4,ae_tb4))
    
    reconstruction = tf.concat([cReconstruction,tReconstruction],1)
    return reconstruction,cReconstruction,tReconstruction
#     return reconstruction

reconstruction,c,t = MDAE_inference(content_x,traffic_x)

all_x = tf.concat([content_x,traffic_x],1) 

loss =  0.5 * tf.reduce_sum(tf.pow(tf.subtract(reconstruction,all_x),2.0))
c_loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(content_x,c),2.0))
t_loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(traffic_x,t),2.0))

#loss =  (0.5 * tf.reduce_sum(tf.pow(tf.subtract(bReconstruction,basic_x),2.0))+0.5 * tf.reduce_sum(tf.pow(tf.subtract(cReconstruction,content_x),2.0))+0.5 * tf.reduce_sum(tf.pow(tf.subtract(tReconstruction,traffic_x),2.0)))/3
#loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(reconstruction,x),2.0))+β*tf.reduce_sum(p * tf.log(p/(tf.reduce_mean(hidden,axis=0)))+ (1-p)*tf.log((1-p)/(1-tf.reduce_mean(hidden,axis=0))))
#+β*tf.reduce_sum(p * tf.log(p/(tf.reduce_mean(hidden,axis=0)))+ (1-p)*tf.log((1-p)/(1-tf.reduce_mean(hidden,axis=0))))+(0.5 * λ) * tf.reduce_sum(tf.reduce_sum(tf.pow(ae_w1,2.0))+tf.reduce_sum(tf.pow(ae1_w2,2.0))+tf.reduce_sum(tf.pow(ae_b1,2.0))+tf.reduce_sum(tf.pow(ae1_b2,2.0)))
#(0.5 * λ) * tf.reduce_sum(tf.reduce_sum(tf.pow(ae_w1,2.0))+tf.reduce_sum(tf.pow(ae1_w2,2.0))+tf.reduce_sum(tf.pow(ae_b1,2.0))+tf.reduce_sum(tf.pow(ae1_b2,2.0)))
train_op = tf.train.GradientDescentOptimizer(learning_rate = 0.003).minimize(loss)
#train_op = tf.train.AdamOptimizer(learning_rate = 0.1).minimize(loss)
#tf.train.AdamOptimizer 
#tf.nn.relu
#tf.nn.tanh

#-----获取随机block数据的函数
def get_random_block_from_data(data2,data3,batch_size):
    start_index = np.random.randint(0,len(data2)-batch_size)
    return data2[start_index:(start_index+batch_size)],data3[start_index:(start_index+batch_size)]

n_samples = int(c_train.shape[0])

#print(n_samples)
training_epochs = 100
batch_size = 100
display_step = 1


#-----训练过程-----
with tf.Session() as sess:
     saver=tf.train.Saver()
     sess.run(tf.global_variables_initializer())
     epoch_list = []
     cost_list = []
     ccost_list = []
     tcost_list = []
     #迭代训练
     for epoch in range(training_epochs):
         avg_cost = 0
         avg_ccost = 0
         avg_tcost = 0
         total_batch = int(n_samples / batch_size)
         for i in range(total_batch):
             batch_c,batch_t = get_random_block_from_data(c_train,t_train,batch_size)
             _,cost,ccost,tcost=sess.run([train_op,loss,c_loss,t_loss], feed_dict={ content_x: batch_c,traffic_x:batch_t})
             avg_cost += (cost / n_samples)*batch_size
             avg_ccost += (ccost / n_samples)*batch_size
             avg_tcost += (tcost / n_samples)*batch_size
         if epoch % display_step == 0:
             print("Epoch",'%04d'%(epoch+1),"cost=","{}".format(avg_cost))
             epoch_list.append(epoch)
             cost_list.append(avg_cost)
             ccost_list.append(avg_ccost)
             tcost_list.append(avg_tcost)
            
     cw1,cb1, tw1,tb1, w2,b2 = sess.run(ae_cw1),sess.run(ae_cb1),sess.run(ae_tw1),sess.run(ae_tb1),sess.run(ae_w2),sess.run(ae_b2)
     
     np.savetxt(cw1Path,cw1,delimiter=",")
     np.savetxt(cb1Path,cb1,delimiter=",")
     np.savetxt(tw1Path,tw1,delimiter=",")
     np.savetxt(tb1Path,tb1,delimiter=",")
     np.savetxt(w2Path,w2,delimiter=",")
     np.savetxt(b2Path,b2,delimiter=",")
     
#     plt.plot(epoch_list,cost_list)
#     plt.axis()
#     plt.xlabel("epochs")
##     plt.savefig("./train_image/SAE_1.png")
#     plt.show()
#     
     fig = plt.figure(figsize=(5,4),dpi = 300)
     plt.ylabel("loss value")
     plt.xlabel("Run number")
     
     plt.xticks([i for i in range(1,training_epochs+1) if i%10 == 0 ])
     plt.yticks([i for i in range(0,training_epochs+1) if i%10 == 0 ])
     
     plt.plot(epoch_list,cost_list,color = 'k',label = "total_loss",ls = '-' ,lw=0.7)
     plt.plot(epoch_list,ccost_list,color = 'c',label = "packet_loss",ls = '--',lw=0.7)
     plt.plot(epoch_list,tcost_list,color = 'b',label = "traffic_loss",ls = '--',lw=0.7)
     #划基准线
#     plt.hlines(0,-1,101,colors='red',linestyles = "dashed",label=u'baseline')
     plt.legend() # 显示图例

     plt.savefig("./fine_tune/loss.png")
     plt.show()


