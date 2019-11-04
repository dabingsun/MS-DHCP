# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 11:02:23 2019

@author: dabing
"""
import tensorflow as tf 
import numpy as np
from loadData import allData 

import matplotlib.pyplot as plt 

#-----读取权重参数路径-----
w1Path = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/singleClassfication/01_fine_tune/traffic/w1.csv'
b1Path = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/singleClassfication/01_fine_tune/traffic/b1.csv'
w2Path = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/singleClassfication/01_fine_tune/traffic/w2.csv'
b2Path = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/singleClassfication/01_fine_tune/traffic/b2.csv'

wsavePath = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/RBMs/trafficRBMPara/43_40_w.csv'
b1savePath = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/RBMs/trafficRBMPara/43_40_b1.csv'
b2savePath = 'E:/workspace for python/Experiment/multimodalExperiment/CICIDS2017/RBMs/trafficRBMPara/43_40_b2.csv'

#-----读取数据-----
allFeatures = allData[:,0:77]

contentFeatures = allData[:,0:34]

trafficFeatures = allData[:,34:77]

X_train = trafficFeatures

#-----参数初始化-----
n_input = 43
n_hidden = 40

#设置占位符
x = tf.placeholder(dtype = tf.float32,shape=[None,n_input],name='x')

#-----自定义参数初始化函数，标准的均匀均匀分布-----
def xavier_init(fan_in,fan_out,constant = 1):
    low = -constant * np.sqrt(6.0/(fan_in + fan_out))
    high = constant * np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype = tf.float32)
#-----读取预训练参数-----

w1 = np.loadtxt(wsavePath, dtype=np.float32, delimiter=",", skiprows=0)
b1 = np.loadtxt(b1savePath, dtype=np.float32, delimiter=",", skiprows=0)
b2 = np.loadtxt(b2savePath, dtype=np.float32, delimiter=",", skiprows=0)
w2 = np.transpose(w1)

#-----当需要共享参数的时候，用get_variable
#def SAE1_inference(x,n_input,n_hidden):
ae_w1 = tf.Variable(w1,name="ae_w1")
ae_b1 = tf.Variable(b1,name = "ae_b1")
ae1_w2 = tf.Variable(w2,name = "ae1_w2")
ae1_b2 =tf.Variable(b2,name="ae1_b2")
hidden = tf.nn.sigmoid(tf.add(tf.matmul(x,ae_w1),ae_b1))
reconstruction = tf.add(tf.matmul(hidden,ae1_w2),ae1_b2)
#     return reconstruction

#reconstruction = SAE1_inference(x,n_input,n_hidden)
loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(reconstruction,x),2.0))
train_op = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)
#tf.train.AdamOptimizer 
#tf.nn.relu
#tf.nn.tanh

#-----获取随机block数据的函数
def get_random_block_from_data(data,batch_size):
    start_index = np.random.randint(0,len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]

n_samples = int(X_train.shape[0])

#print(n_samples)
training_epochs = 100
batch_size = 50
display_step = 1


#-----训练过程-----
with tf.Session() as sess:
     saver=tf.train.Saver()
     sess.run(tf.global_variables_initializer())
     epoch_list = []
     cost_list = []
     #迭代训练
     for epoch in range(training_epochs):
         avg_cost = 0
         total_batch = int(n_samples / batch_size)
         for i in range(total_batch):
             batch_x = get_random_block_from_data(X_train,batch_size)
             _,cost=sess.run([train_op,loss], feed_dict={x: batch_x})
             avg_cost += (cost / n_samples)*batch_size
         if epoch % display_step == 0:
             print("Epoch",'%04d'%(epoch+1),"cost=","{}".format(avg_cost))
             epoch_list.append(epoch)
             cost_list.append(avg_cost)
            
     w1, b1, w2, b2 = sess.run(ae_w1),sess.run(ae_b1),sess.run(ae1_w2),sess.run(ae1_b2)
     
     np.savetxt(w1Path,w1,delimiter=",")
     np.savetxt(b1Path,b1,delimiter=",")
     np.savetxt(w2Path,w2,delimiter=",")
     np.savetxt(b2Path,b2,delimiter=",")
     
     plt.plot(epoch_list,cost_list)
     plt.axis()
     plt.xlabel("epochs")
#     plt.savefig("./train_image/SAE_1.png")
     plt.show()



