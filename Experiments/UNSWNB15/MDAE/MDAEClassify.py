# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:17:04 2019

@author: dabing
"""

import tensorflow as tf 
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from loadData import trainbasicFeatures,traincontentFeatures,traintrafficFeatures,trainLabels_2,testbasicFeatures,testcontentFeatures,testtrafficFeatures,testLabels_2 
import time
#-----读取权重参数路径-----
bw1Path = 'E:/workspace for python/Experiment/multimodalExperiment/UNSWNB15/MDAE/fine_tune/120_30bw1.csv'
bb1Path = 'E:/workspace for python/Experiment/multimodalExperiment/UNSWNB15/MDAE/fine_tune/120_30bb1.csv'

cw1Path = 'E:/workspace for python/Experiment/multimodalExperiment/UNSWNB15/MDAE/fine_tune/120_30cw1.csv'
cb1Path = 'E:/workspace for python/Experiment/multimodalExperiment/UNSWNB15/MDAE/fine_tune/120_30cb1.csv'

tw1Path = 'E:/workspace for python/Experiment/multimodalExperiment/UNSWNB15/MDAE/fine_tune/120_30tw1.csv'
tb1Path = 'E:/workspace for python/Experiment/multimodalExperiment/UNSWNB15/MDAE/fine_tune/120_30tb1.csv'

w2Path = 'E:/workspace for python/Experiment/multimodalExperiment/UNSWNB15/MDAE/fine_tune/120_30w2.csv'
b2Path = 'E:/workspace for python/Experiment/multimodalExperiment/UNSWNB15/MDAE/fine_tune/120_30b2.csv'
#-----读取数据-----
b_train,c_train,t_train = trainbasicFeatures,traincontentFeatures,traintrafficFeatures
b_test,c_test, t_test = testbasicFeatures,testcontentFeatures,testtrafficFeatures
train_y,test_y = trainLabels_2,testLabels_2 

bw1 = np.loadtxt(bw1Path, dtype=np.float32, delimiter=",", skiprows=0)
bb1 = np.loadtxt(bb1Path, dtype=np.float32, delimiter=",", skiprows=0)

cw1 = np.loadtxt(cw1Path, dtype=np.float32, delimiter=",", skiprows=0)
cb1 = np.loadtxt(cb1Path, dtype=np.float32, delimiter=",", skiprows=0)

tw1 = np.loadtxt(tw1Path, dtype=np.float32, delimiter=",", skiprows=0)
tb1 = np.loadtxt(tb1Path, dtype=np.float32, delimiter=",", skiprows=0)


w2 = np.loadtxt(w2Path, dtype=np.float32, delimiter=",", skiprows=0)
b2 = np.loadtxt(b2Path, dtype=np.float32, delimiter=",", skiprows=0)

ae_bw1 = tf.Variable(bw1,name="ae_bw1")
ae_bb1 = tf.Variable(bb1,name = "ae_bb1")

ae_cw1 = tf.Variable(cw1,name="ae_cw1")
ae_cb1 = tf.Variable(cb1,name = "ae_cb1")

ae_tw1 = tf.Variable(tw1,name="ae_tw1")
ae_tb1 = tf.Variable(tb1,name = "ae_tb1")

ae_w2 = tf.Variable(w2,name = "ae_w2")
ae_b2 =tf.Variable(b2,name="ae_b2")

n_hidden = 30
n_output = 2

ae_w3 = tf.Variable(tf.zeros([n_hidden,n_output],dtype=tf.float32),name = "ae_w3")
ae_b3 =tf.Variable(tf.zeros([n_output],dtype=tf.float32),name="ae_b3")

basic_x =tf.placeholder(dtype = tf.float32,shape=[None,169],name='basic_x')
content_x =tf.placeholder(dtype = tf.float32,shape=[None,15],name='content_x')
traffic_x =tf.placeholder(dtype = tf.float32,shape=[None,12],name='traffic_x')
y_ = tf.placeholder(dtype=tf.float32,shape=[None,n_output],name="y_")

def MDAE_inference(basic_x,content_x,traffic_x):
    
    bHidden1 = tf.nn.sigmoid(tf.add(tf.matmul(basic_x,ae_bw1),ae_bb1))
    cHidden1 = tf.nn.sigmoid(tf.add(tf.matmul(content_x,ae_cw1),ae_cb1))
    tHidden1 = tf.nn.sigmoid(tf.add(tf.matmul(traffic_x,ae_tw1),ae_tb1))
    hidden1 = tf.concat([bHidden1,cHidden1,tHidden1],1)
    
    hidden2 = tf.nn.sigmoid(tf.add(tf.matmul(hidden1,ae_w2),ae_b2))
    
    logists = tf.nn.softmax(tf.add(tf.matmul(hidden2,ae_w3),ae_b3))
    return logists

y = MDAE_inference(basic_x,content_x,traffic_x)

loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)+(1-y_)*tf.log(1-y),axis=1))

train_op = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss)
#train_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

y_pre = tf.argmax(y,1)

#-----获取随机block数据的函数
def get_random_block_from_data(data1,data2,data3,data_y,batch_size):
    start_index = np.random.randint(0,len(data1)-batch_size)
    return data1[start_index:(start_index+batch_size)],data2[start_index:(start_index+batch_size)],data3[start_index:(start_index+batch_size)],data_y[start_index:(start_index+batch_size)]
#按照batch取数据
def get_data(data_x, data_y, batch_size, index):
    return data_x[index:(index+batch_size)],data_y[index:(index+batch_size)]

n_samples = int(trainbasicFeatures.shape[0])

training_epochs = 10
batch_size = 100
display_step = 1

#-----训练过程-----
with tf.Session() as sess:
     saver=tf.train.Saver()
     sess.run(tf.global_variables_initializer())
     epoch_list = []
     train_cost_list = []
     train_ac_list = []
     test_cost_list = []
     test_ac_list = []
     cm_list = []
     trainTime = []
     testTime = []
     #迭代训练
     for epoch in range(training_epochs):
         total_cost = 0
         total_ac = 0
         total_test_ac = 0
         total_test_cost = 0
         
         total_batch = int(n_samples / batch_size)
         trainS = time.time()
         for i in range(total_batch):
             batch_b,batch_c,batch_t,batch_y = get_random_block_from_data(b_train,c_train,t_train,train_y,batch_size)
             _, cost, ac =sess.run([train_op,loss,acc], feed_dict={basic_x: batch_b, content_x: batch_c,traffic_x:batch_t, y_:batch_y})
             total_ac += ac
             total_cost += cost
         trainT = time.time()-trainS
         teatS = time.time()
         y_t = np.argmax(test_y,1)
         total_test_cost,total_test_ac,y_p = sess.run([loss,acc,y_pre], feed_dict={basic_x: b_test, content_x: c_test,traffic_x:t_test, y_:test_y})
         testT = time.time() - teatS
         cm = confusion_matrix(y_t, y_p)
         
         if epoch % display_step == 0:
             print("Epoch",'%04d'%(epoch+1),
                   "[ train_ac={0:.4f} train_cost={1:.4f} ] [ test_ac={2:.4f} test_cost={3:.4f} ]".format(
                           total_ac/total_batch,
                           total_cost/total_batch,
                           total_test_ac,
                           total_test_cost))
             epoch_list.append(epoch)
             train_cost_list.append(total_cost/total_batch)
             train_ac_list.append(total_ac/total_batch)
             test_ac_list.append(total_test_ac)
             test_cost_list.append(total_test_cost)
             cm_list.append(cm)
             trainTime.append(trainT)
             testTime.append(testT)
     print('avg_train_time:{0}s,avg_train_time:{1}s '.format(sum(trainTime)*10,sum(testTime)*10))
#     np.savetxt('./accuracytrain.csv',train_ac_list,delimiter=",")
#     np.savetxt('./accuracytest.csv',test_ac_list,delimiter=",")
#     创建两行一列的图框，并获得第一行一个区域
     ax1 = plt.subplot(2,1,1)
     plt.plot(epoch_list,train_ac_list,color = 'green',label = "Train_acc")
     plt.plot(epoch_list,test_ac_list,color = 'red',label = "Test+_acc")
     plt.axis()
     plt.legend()
     plt.ylabel("accuracy")
     plt.xlabel("epochs")
     ax1.set_title("accuracy-epochs")
     
     ax2 = plt.subplot(2,1,2)
     plt.plot(epoch_list,train_cost_list,color = 'green',label = "Train_cost")
     plt.plot(epoch_list,test_cost_list,color = 'red',label = "Test+_cost")
     plt.axis()
     plt.legend()
     plt.ylabel("loss")
     plt.xlabel("epochs")
     ax2.set_title("loss-epochs")
#     plt.savefig("./train_image/SAE_1.png")
     plt.tight_layout()
     plt.show()
     
     print("maxAcc {}".format(max(test_ac_list)))
     print("CM:{0}".format(cm_list[test_ac_list.index(max(test_ac_list))]))
     print("lastCM:{0}".format(cm_list[-1]))

