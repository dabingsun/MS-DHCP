# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 15:19:30 2019

@author: dabing
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 16:13:55 2019

@author: dabing
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 11:01:54 2019

@author: dabing
"""

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from loadData import trainFeatures,trainLabels_2,testFeatures,testLabels_2


#-----读取权重参数路径-----

w1Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/noModalityClaassification/fine_tune/w1.csv'
b1Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/noModalityClaassification/fine_tune/b1.csv'
#w2Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/singleClassfication/fine_tune/basic/w2.csv'
#b2Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/singleClassfication/fine_tune/basic/b2.csv'



#-----读取数据-----
train_x = trainFeatures
train_y = trainLabels_2

test_x = testFeatures
test_y = testLabels_2

#-----参数初始化-----
n_input = 122
n_hidden = 60
n_output = 2
#设置占位符
x = tf.placeholder(dtype = tf.float32,shape=[None,n_input],name='x')
y_ = tf.placeholder(dtype=tf.float32,shape=[None,n_output],name="y_")
#-----自定义参数初始化函数，标准的均匀均匀分布-----
def xavier_init(fan_in,fan_out,constant = 1):
    low = -constant * np.sqrt(6.0/(fan_in + fan_out))
    high = constant * np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype = tf.float32)
#-----读取预训练参数-----

w1 = np.loadtxt(w1Path, dtype=np.float32, delimiter=",", skiprows=0)
b1 = np.loadtxt(b1Path, dtype=np.float32, delimiter=",", skiprows=0)
#b2 = np.loadtxt(b2savePath, dtype=np.float32, delimiter=",", skiprows=0)
#w2 = np.transpose(w1)

#-----当需要共享参数的时候，用get_variable
#def SAE1_inference(x,n_input,n_hidden):

ae_w1 = tf.Variable(w1,name="ae_w1")
ae_b1 = tf.Variable(b1,name = "ae_b1")
ae1_w2 = tf.Variable(tf.zeros([n_hidden,n_output],dtype=tf.float32), name = "ae1_w2")
ae1_b2 =tf.Variable(tf.zeros([n_output],dtype=tf.float32), name="ae1_b2")
hidden = tf.nn.sigmoid(tf.add(tf.matmul(x,ae_w1),ae_b1))
#hidden = tf.nn.relu (tf.add(tf.matmul(x,ae_w1),ae_b1))
logists = tf.nn.softmax(tf.add(tf.matmul(hidden,ae1_w2),ae1_b2))
y = logists

loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)+(1-y_)*tf.log(1-y),axis=1))

#loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(reconstruction,x),2.0))
#train_op = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)
train_op = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss)
#tf.train.AdamOptimizer 
#tf.nn.relu
#tf.nn.tanh
#定义训练精度
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#tf.cast 将之前bool值转为float32,reduce_mean求平均
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

y_pre = tf.argmax(y,1)
#-----获取随机block数据的函数
def get_random_block_from_data(data_x,data_y,batch_size):
    start_index = np.random.randint(0,len(data_x)-batch_size)
    return data_x[start_index:(start_index+batch_size)], data_y[start_index:(start_index+batch_size)]

#按照batch取数据
def get_data(data_x, data_y, batch_size, index):
    return data_x[index*batch_size:(index+1)*batch_size],data_y[index*batch_size:(index+1)*batch_size]

n_samples = int(train_x.shape[0])

#print(n_samples)
training_epochs = 100
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
     #迭代训练
     for epoch in range(training_epochs):
         total_cost = 0
         total_ac = 0
         total_test_ac = 0
         total_test_cost = 0
         
         total_batch = int(n_samples / batch_size)
         for i in range(total_batch):
             batch_x, batch_y = get_random_block_from_data(train_x, train_y,batch_size)
#             batch_x, batch_y = get_data(train_x, train_y,batch_size,i)
             _, cost, ac =sess.run([train_op,loss,acc], feed_dict={x: batch_x, y_:batch_y})
             total_ac += ac
             total_cost += cost
             
         y_t = np.argmax(test_y,1)
         total_test_cost,total_test_ac,y_p = sess.run([loss,acc,y_pre], feed_dict={x: test_x, y_:test_y})
         
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


