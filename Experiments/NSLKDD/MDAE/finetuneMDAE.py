# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 10:09:15 2019

@author: dabing
"""

import tensorflow as tf 
import numpy as np
from loadData import trainbasicFeatures,traincontentFeatures,traintrafficFeatures 

import matplotlib.pyplot as plt 


#-----读取权重参数路径-----
bwloadPath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/basicRBMPara/90_60_w.csv'
bb1loadPath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/basicRBMPara/90_60_b1.csv'
bb2loadPath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/basicRBMPara/90_60_b2.csv'

cwloadPath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/contentRBMPara/13_10_w.csv'
cb1loadPath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/contentRBMPara/13_10_b1.csv'
cb2loadPath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/contentRBMPara/13_10_b2.csv'

twloadPath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/trafficRBMPara/19_15_w.csv'
tb1loadPath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/trafficRBMPara/19_15_b1.csv'
tb2loadPath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/trafficRBMPara/19_15_b2.csv'

w2loadPath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/multiDBNPara/95_20_w.csv'
b2loadPath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/multiDBNPara/95_20_b1.csv'
b3loadPath = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/RBMs/multiDBNPara/95_20_b2.csv'

#-----保存权重参数路径-----
bw1Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/MDAE/fine_tune/95_20bw1.csv'
bb1Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/MDAE/fine_tune/95_20bb1.csv'

cw1Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/MDAE/fine_tune/95_20cw1.csv'
cb1Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/MDAE/fine_tune/95_20cb1.csv'

tw1Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/MDAE/fine_tune/95_20tw1.csv'
tb1Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/MDAE/fine_tune/95_20tb1.csv'

w2Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/MDAE/fine_tune/95_20w2.csv'
b2Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/MDAE/fine_tune/95_20b2.csv'


#-----读取数据-----
b_train,c_train,t_train = trainbasicFeatures,traincontentFeatures,traintrafficFeatures 

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

bw1 = np.loadtxt(bwloadPath, dtype=np.float32, delimiter=",", skiprows=0)
bb1 = np.loadtxt(bb1loadPath, dtype=np.float32, delimiter=",", skiprows=0)
bb4 = np.loadtxt(bb2loadPath, dtype=np.float32, delimiter=",", skiprows=0)
#bw4 = np.transpose(bw1)

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



ae_bw1 = tf.Variable(bw1,name="ae_bw1")
ae_bb1 = tf.Variable(bb1,name = "ae_bb1")
#ae_bw4 = tf.Variable(bw4,name="ae_bw4")
ae_bb4 = tf.Variable(bb4,name = "ae_bb4")

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

basic_x =tf.placeholder(dtype = tf.float32,shape=[None,90],name='basic_x')
content_x =tf.placeholder(dtype = tf.float32,shape=[None,13],name='content_x')
traffic_x =tf.placeholder(dtype = tf.float32,shape=[None,19],name='traffic_x')
#basic_x content_x traffic_x
def MDAE_inference(basic_x,content_x,traffic_x):
    
    bHidden1 = tf.nn.sigmoid(tf.add(tf.matmul(basic_x,ae_bw1),ae_bb1))
    cHidden1 = tf.nn.sigmoid(tf.add(tf.matmul(content_x,ae_cw1),ae_cb1))
    tHidden1 = tf.nn.sigmoid(tf.add(tf.matmul(traffic_x,ae_tw1),ae_tb1))
    hidden1 = tf.concat([bHidden1,cHidden1,tHidden1],1)
    
    hidden2 = tf.nn.sigmoid(tf.add(tf.matmul(hidden1,ae_w2),ae_b2))
    
    hidden3 = tf.nn.sigmoid(tf.add(tf.matmul(hidden2,tf.transpose(ae_w2)),ae_b3))
#    hidden3 = tf.add(tf.matmul(hidden2,ae_w3),ae_b3)
    
    bHidden3 = tf.slice(hidden3,[0,0],[-1,70])
    cHidden3 = tf.slice(hidden3,[0,70],[-1,10])
    tHidden3 = tf.slice(hidden3,[0,80],[-1,-1])
    

    bReconstruction = tf.nn.sigmoid(tf.add(tf.matmul(bHidden3,tf.transpose(ae_bw1)),ae_bb4))
    cReconstruction = tf.nn.sigmoid(tf.add(tf.matmul(cHidden3,tf.transpose(ae_cw1)),ae_cb4))
    tReconstruction = tf.nn.sigmoid(tf.add(tf.matmul(tHidden3,tf.transpose(ae_tw1)),ae_tb4))
#    bReconstruction = tf.add(tf.matmul(bHidden3,ae_bw4,ae_bb4))
#    cReconstruction = tf.add(tf.matmul(cHidden3,ae_cw4,ae_cb4))
#    tReconstruction = tf.add(tf.matmul(tHidden3,ae_tw4,ae_tb4))
    
    reconstruction = tf.concat([bReconstruction,cReconstruction,tReconstruction],1)
    return reconstruction,bReconstruction,cReconstruction,tReconstruction
#     return reconstruction

reconstruction,b,c,t = MDAE_inference(basic_x,content_x,traffic_x)

all_x = tf.concat([basic_x,content_x,traffic_x],1) 

loss =  0.5 * tf.reduce_sum(tf.pow(tf.subtract(reconstruction,all_x),2.0))
b_loss =  0.5 * tf.reduce_sum(tf.pow(tf.subtract(basic_x,b),2.0))
c_loss =  0.5 * tf.reduce_sum(tf.pow(tf.subtract(content_x,c),2.0))
t_loss =  0.5 * tf.reduce_sum(tf.pow(tf.subtract(traffic_x,t),2.0))


#loss =  (0.5 * tf.reduce_sum(tf.pow(tf.subtract(bReconstruction,basic_x),2.0))+0.5 * tf.reduce_sum(tf.pow(tf.subtract(cReconstruction,content_x),2.0))+0.5 * tf.reduce_sum(tf.pow(tf.subtract(tReconstruction,traffic_x),2.0)))/3
#loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(reconstruction,x),2.0))+β*tf.reduce_sum(p * tf.log(p/(tf.reduce_mean(hidden,axis=0)))+ (1-p)*tf.log((1-p)/(1-tf.reduce_mean(hidden,axis=0))))
#+β*tf.reduce_sum(p * tf.log(p/(tf.reduce_mean(hidden,axis=0)))+ (1-p)*tf.log((1-p)/(1-tf.reduce_mean(hidden,axis=0))))+(0.5 * λ) * tf.reduce_sum(tf.reduce_sum(tf.pow(ae_w1,2.0))+tf.reduce_sum(tf.pow(ae1_w2,2.0))+tf.reduce_sum(tf.pow(ae_b1,2.0))+tf.reduce_sum(tf.pow(ae1_b2,2.0)))
#(0.5 * λ) * tf.reduce_sum(tf.reduce_sum(tf.pow(ae_w1,2.0))+tf.reduce_sum(tf.pow(ae1_w2,2.0))+tf.reduce_sum(tf.pow(ae_b1,2.0))+tf.reduce_sum(tf.pow(ae1_b2,2.0)))
train_op = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss)
#train_op = tf.train.AdamOptimizer(learning_rate = 0.1).minimize(loss)
#tf.train.AdamOptimizer 
#tf.nn.relu
#tf.nn.tanh

#-----获取随机block数据的函数
def get_random_block_from_data(data1,data2,data3,batch_size):
    start_index = np.random.randint(0,len(data1)-batch_size)
    return data1[start_index:(start_index+batch_size)],data2[start_index:(start_index+batch_size)],data3[start_index:(start_index+batch_size)]

n_samples = int(b_train.shape[0])

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
     bcost_list = []
     ccost_list = []
     tcost_list = []
     #迭代训练
     for epoch in range(training_epochs):
         avg_cost = 0
         avg_bcost = 0
         avg_ccost = 0
         avg_tcost = 0
         total_batch = int(n_samples / batch_size)
         for i in range(total_batch):
             batch_b,batch_c,batch_t = get_random_block_from_data(b_train,c_train,t_train,batch_size)
             _,cost,bcost,ccost,tcost=sess.run([train_op,loss,b_loss,c_loss,t_loss], feed_dict={basic_x: batch_b, content_x: batch_c,traffic_x:batch_t})
             avg_cost += (cost / n_samples)*batch_size
             avg_bcost += (bcost / n_samples)*batch_size
             avg_ccost += (ccost / n_samples)*batch_size
             avg_tcost += (tcost / n_samples)*batch_size
         if epoch % display_step == 0:
             print("Epoch",'%04d'%(epoch+1),"cost=","{}".format(avg_cost))
             epoch_list.append(epoch)
             cost_list.append(avg_cost)
             bcost_list.append(avg_bcost)
             ccost_list.append(avg_ccost)
             tcost_list.append(avg_tcost)
            
     bw1,bb1, cw1,cb1, tw1,tb1, w2,b2 = sess.run(ae_bw1),sess.run(ae_bb1),sess.run(ae_cw1),sess.run(ae_cb1),sess.run(ae_tw1),sess.run(ae_tb1),sess.run(ae_w2),sess.run(ae_b2)
     
     np.savetxt(bw1Path,bw1,delimiter=",")
     np.savetxt(bb1Path,bb1,delimiter=",")
     np.savetxt(cw1Path,cw1,delimiter=",")
     np.savetxt(cb1Path,cb1,delimiter=",")
     np.savetxt(tw1Path,tw1,delimiter=",")
     np.savetxt(tb1Path,tb1,delimiter=",")
     np.savetxt(w2Path,w2,delimiter=",")
     np.savetxt(b2Path,b2,delimiter=",")
     
     fig = plt.figure(figsize=(5,4),dpi = 300)
     plt.ylabel("loss value")
     plt.xlabel("Run number")
     
     plt.xticks([i for i in range(1,training_epochs+1) if i%10 == 0 ])
#     plt.yticks([i for i in range(0,220) if i%20 == 0 ])
     
     plt.plot(epoch_list,cost_list,c = 'k', label='total_loss',ls='-',lw=0.8)
     plt.plot(epoch_list,bcost_list,c = 'b', label='basic_loss',ls='--',lw=0.8)
     plt.plot(epoch_list,ccost_list,c = 'g', label='content_loss',ls='--',lw=0.8)
     plt.plot(epoch_list,tcost_list,c = 'c', label='traffic_loss',ls='--',lw=0.8)
     plt.legend() # 显示图例

     plt.savefig("./fine_tune/loss.png")
     plt.show()


