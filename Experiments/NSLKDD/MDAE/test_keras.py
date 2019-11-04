
import tensorflow as tf
import numpy as np
from loadData import trainbasicFeatures,traincontentFeatures,traintrafficFeatures,trainLabels_5,testbasicFeatures,testcontentFeatures,testtrafficFeatures,testLabels_5 

#-----读取权重参数路径-----
bw1Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/MDAE/fine_tune/95_60bw1.csv'
bb1Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/MDAE/fine_tune/95_60bb1.csv'

cw1Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/MDAE/fine_tune/95_60cw1.csv'
cb1Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/MDAE/fine_tune/95_60cb1.csv'

tw1Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/MDAE/fine_tune/95_60tw1.csv'
tb1Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/MDAE/fine_tune/95_60tb1.csv'

w2Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/MDAE/fine_tune/95_60w2.csv'
b2Path = 'E:/workspace for python/Experiment/multimodalExperiment/NSLKDD/MDAE/fine_tune/95_60b2.csv'

bw1 = np.loadtxt(bw1Path, dtype=np.float32, delimiter=",", skiprows=0)
bb1 = np.loadtxt(bb1Path, dtype=np.float32, delimiter=",", skiprows=0)

cw1 = np.loadtxt(cw1Path, dtype=np.float32, delimiter=",", skiprows=0)
cb1 = np.loadtxt(cb1Path, dtype=np.float32, delimiter=",", skiprows=0)

tw1 = np.loadtxt(tw1Path, dtype=np.float32, delimiter=",", skiprows=0)
tb1 = np.loadtxt(tb1Path, dtype=np.float32, delimiter=",", skiprows=0)


w2 = np.loadtxt(w2Path, dtype=np.float32, delimiter=",", skiprows=0)
b2 = np.loadtxt(b2Path, dtype=np.float32, delimiter=",", skiprows=0)

#b_train,c_train,t_train = trainbasicFeatures,traincontentFeatures,traintrafficFeatures
#b_test,c_test, t_test = testbasicFeatures,testcontentFeatures,testtrafficFeatures
#train_y,test_y = trainLabels_5,testLabels_5 

input_b = tf.keras.Input((90,),name='input_b')
hidden_b = tf.keras.layers.Dense(70, activation='sigmoid',use_bias = True, kernel_initializer =tf.keras.initializers.Constant(bw1), bias_initializer=tf.keras.initializers.Constant(bb1))(input_b)

input_c = tf.keras.Input((13,),name='input_c')
hidden_c = tf.keras.layers.Dense(10, activation='sigmoid',use_bias = True, kernel_initializer =tf.keras.initializers.Constant(cw1), bias_initializer=tf.keras.initializers.Constant(cb1))(input_c)

input_t = tf.keras.Input((19,),name='input_t')
hidden_t = tf.keras.layers.Dense(15, activation='sigmoid',use_bias = True, kernel_initializer =tf.keras.initializers.Constant(tw1), bias_initializer=tf.keras.initializers.Constant(tb1))(input_t)

merged_vector = tf.keras.layers.concatenate([hidden_b,hidden_c,hidden_t],axis=1)

hidden_all = tf.keras.layers.Dense(60,activation='sigmoid',use_bias = True, kernel_initializer =tf.keras.initializers.Constant(w2), bias_initializer=tf.keras.initializers.Constant(b2))(merged_vector)

output = tf.keras.layers.Dense(5,activation='softmax',use_bias = True)(hidden_all)

#restruction_merged_vector = tf.keras.layers.Dense(95,activation='sigmoid',use_bias=True)(hidden_all)
#
#restruction_hidden_b = tf.slice(restruction_merged_vector,[0,0],[-1,70])
#restruction_hidden_c = tf.slice(restruction_merged_vector,[0,70],[-1,10])
#restruction_hidden_t = tf.slice(restruction_merged_vector,[0,80],[-1,-1])
#
#restruction_b = tf.keras.layers.Dense(90,use_bias=True,name='restruction_b')(restruction_hidden_b)
#restruction_c = tf.keras.layers.Dense(13,use_bias=True,name='restruction_c')(restruction_hidden_c)
#restruction_t = tf.keras.layers.Dense(19,use_bias=True,name='restruction_t')(restruction_hidden_t)

#model = tf.keras.Model(inputs=[input_b, input_c, input_t], outputs=[restruction_b, restruction_c, restruction_t])
model = tf.keras.Model(inputs=[input_b, input_c, input_t], outputs=[output])
model.summary()
model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#model.fit([trainbasicFeatures, traincontentFeatures,traintrafficFeatures], [trainbasicFeatures, traincontentFeatures,traintrafficFeatures], 
#          epochs=100,batch_size=100)

model.fit([trainbasicFeatures, traincontentFeatures,traintrafficFeatures], [trainLabels_5], 
          epochs=100,batch_size=50)
acc0=model.evaluate([testbasicFeatures,testcontentFeatures,testtrafficFeatures], testLabels_5,verbose=0)
print(acc0)