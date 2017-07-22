# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 10:10:50 2017

@author: O222069
"""

import numpy as np
import pandas as pd
import tflearn

#download the titanic dataset
from tflearn.datasets import titanic
data=pd.read_csv(titanic.download_dataset('titanic_dataset.csv'))

#load csv file
from tflearn.data_utils import load_csv,samplewise_std_normalization
data,labels=load_csv('titanic_dataset.csv',target_column=0,categorical_labels=True,n_classes=2)

#preprocessing function

def preprocess(data,columns_to_ignore):
    for id in sorted(columns_to_ignore,reverse=True):
        [r.pop(id) for r in data]
    for i in range(len(data)):
        data[i][1]=1. if data[i][1]=='female' else 0.
    return np.array(data,dtype=np.float32)

to_ignore=[1,6]

data=preprocess(data,to_ignore)

#data=samplewise_std_normalization(data)

from sklearn.preprocessing import StandardScaler,MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
data=sc.fit_transform(data)

#build neural network
net=tflearn.input_data(shape=[None,6])
net=tflearn.fully_connected(net,32)
net=tflearn.fully_connected(net,32)
net=tflearn.fully_connected(net,2,activation='softmax')
net=tflearn.regression(net)

#define model
model=tflearn.DNN(net)
#start training
model.fit(data,labels,n_epoch=100,batch_size=16,show_metric=True)

#test
dicaprio=[3,'Jack','male',19,0,0,'N/A',5.000]

winslet=[1,'Rose','female',17,1,2,'N/A',100.000]
#preprocess
dicaprio,winslet=preprocess([dicaprio,winslet],to_ignore)
test=np.array([dicaprio,winslet])
sc.fit(test)

pred=model.predict(test)
print("Dicaprio Survivng Rate:",pred[0][1])
print("Winslet Survivng Rate:",pred[1][1])

import tflearn
#linear regression
X = [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1]
Y = [1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3]

inputt=tflearn.input_data(shape=[None])
linear=tflearn.single_unit(inputt)
reg=tflearn.regression(linear,optimizer='sgd',loss='mean_square',learning_rate=0.01)
m=tflearn.DNN(reg)
m.fit(X,Y,n_epoch=100,show_metric=True,snapshot_epoch=False)
print("\nRegression Results:")
print("Y="+ str(m.get_weights(linear.W))+"*X+"+ str(m.get_weights(linear.b)))
print("\nTest Prediction for x=3.2,3.3,3.4")
print(m.predict([3.2,3.3,3.4]))
