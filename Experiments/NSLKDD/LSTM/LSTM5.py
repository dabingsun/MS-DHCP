# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 08:12:12 2019

@author: dabing
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import time as run_time
from sklearn.metrics import confusion_matrix
from loadData import trainFeatures,trainLabels_5,testFeatures,testLabels_5 

start_time = run_time.time()
#参数初始化
rnn_input = 122
categories_num = 5

batch_size = 25
time_length= 500

rnn_layer = 1
rnn_hidden = 80
lr = 0.5

x_train = trainFeatures
x_test = testFeatures
train_y,test_y = trainLabels_5,testLabels_5 

w_out = tf.get_variable("w_out",shape = [rnn_hidden,categories_num],dtype=tf.float32,initializer=tf.zeros_initializer(dtype=tf.float32))
b_out = tf.get_variable("b_out",shape=[categories_num],dtype=tf.float32,initializer=tf.zeros_initializer(dtype=tf.float32))
#自定义参数初始化函数，标准的均匀均匀分布
def xavier_init(fan_in,fan_out,constant = 1):
    low = -constant * np.sqrt(6.0/(fan_in + fan_out))
    high = constant * np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype = tf.float32)

class RNNModel(object):
    def __init__(self,is_train, batch_size, time_steps):
#        tf.reset_default_graph()
        self.is_train = is_train 
        self.batch_size = batch_size
        
        self.x =tf.placeholder(dtype = tf.float32,shape=[None,rnn_input],name='x')
        
        self.y_ = tf.placeholder(dtype=tf.float32,shape=[None,categories_num],name="y_")
        
        self.rnn_input = tf.reshape(self.x,shape =(-1,time_steps,rnn_input))
        
        if self.is_train:
            self.temp_cell = ([tf.contrib.rnn.DropoutWrapper(self.get_cell(),output_keep_prob=1) for i in range(rnn_layer)])
        else:
            self.temp_cell = ([self.get_cell() for i in range(rnn_layer)])
        
        self.cell  = tf.nn.rnn_cell.MultiRNNCell(self.temp_cell,state_is_tuple=True)
        self.is_static = self.cell.zero_state(batch_size,dtype=tf.float32)
        self.output,self.static = tf.nn.dynamic_rnn(self.cell,self.rnn_input,initial_state=self.is_static)
    
    #将RNN的输出数据转为softmax的输入数据（三维转二维）
        self.rnn_output = tf.reshape(tf.concat(self.output,1),(-1,rnn_hidden))
            
        self.logists = tf.nn.softmax(tf.add(tf.matmul(self.rnn_output,w_out),b_out))
        #定义损失函数
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.logists)+(1-self.y_)*tf.log(1-self.logists),axis=1))
        #定义优化函数及学习率
#        self.train_op = tf.train.AdamOptimizer(learning_rate=lr ).minimize(self.loss)
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(self.loss)
        #定义训练精度
        self.correct_prediction = tf.equal(tf.argmax(self.logists,1),tf.argmax(self.y_,1))
        #tf.cast 将之前bool值转为float32,reduce_mean求平均
        self.acc= tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.y_p = tf.argmax(self.logists,1)
    def get_cell(self):
        return tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_hidden,activation = tf.nn.sigmoid)

#训练时随机获取batch数据
def get_bolck_data(data, data_y, time_length, batch_size):
    end_random_index = (len(data) // time_length) - batch_size
    start_index = np.random.randint(0,end_random_index)
    return data[start_index*time_length:(start_index+batch_size)*time_length],data_y[start_index*time_length:(start_index+batch_size)*time_length]

#数据集的迭代次数
epochs =100 
display = 1
n_samples = int(trainFeatures.shape[0])
#调用读取数据模块的数据



with tf.Session() as sess:
    with tf.variable_scope("model",reuse=None):
        train_model = RNNModel(False,batch_size=batch_size, time_steps=time_length)
    with tf.variable_scope("model",reuse=True):
        test_model = RNNModel(False, batch_size=1, time_steps= x_test.shape[0])
        
    sess.run(tf.global_variables_initializer())
    epoch_list = []
    train_cost_list = []
    train_ac_list = []
    test_cost_list = []
    test_ac_list = []
    cm_list = []
    
    for epoch in range(epochs):
        steps = 0
        total_cost = 0
        total_ac = 0
        total_test_ac = 0
        total_test_cost = 0
#        train_x.shape[0]// (batch_size*time_length//2)
        total_batch = int(n_samples / (batch_size*time_length))
        for step in range(total_batch):
            batch_x,batch_y = get_bolck_data(x_train,train_y,time_length,batch_size)
#            x_train_a, y_train_a = get_data(train_x, train_y, time_length, batch_size, step)
            batch_y = batch_y.reshape(-1,categories_num)
            _, cost, ac = sess.run([train_model.train_op,train_model.loss,train_model.acc],
                                   feed_dict={train_model.x:batch_x, train_model.y_:batch_y})
            steps += 1
            total_ac += ac
            total_cost += cost
            
    #    #test+测试结果
        test_y = test_y.reshape(-1,categories_num)
        y_t = np.argmax(test_y,1)
        total_test_cost,total_test_ac,y_p = sess.run([test_model.loss,test_model.acc,test_model.y_p],
                                                     feed_dict={test_model.x:x_test, test_model.y_:test_y})
        cm = confusion_matrix(y_t, y_p)
        
        if epoch % display == 0:
             print("Epoch",'%04d'%(epoch+1),
                   "[ train_ac={0:.4f} train_cost={1:.4f} ] [ test_ac={2:.4f} test_cost={3:.4f} ]".format(
                           total_ac/steps,
                           total_cost/steps,
                           total_test_ac,
                           total_test_cost))
             epoch_list.append(epoch)
             train_cost_list.append(total_cost/steps)
             train_ac_list.append(total_ac/steps)
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







