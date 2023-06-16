# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 12:40:05 2023

@author: user
"""
# Load packages
import cv2
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm 

import keras 
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import accuracy_score

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Path set
root_path = 'C:\\Hu\\0_IMIS\\5_course\\DeepLearning\\'
data_path = root_path + 'data\\'

# Split required data and decide model input size 
train, test, val = ([],[],[])
train_images, train_labels, test_images, test_labels, val_images, val_labels = ([],[],[],[],[],[])
data_type = ['train', 'test', 'val'] # ['train', 'test', 'val']
column = 1
size = 64

# Read category from txt file
for i in range(len(data_type)):
    globals()[data_type[i]] = np.genfromtxt(data_path + data_type[i] + '.txt', delimiter=' ', dtype = 'str', skip_header=1)
    # globals()[data_type[i]] = list(filter(lambda x: int(x[1]) <= 4, globals()[data_type[i]]))    


def colorQuant(img, Z, K, criteria):

   ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
   
   # Now convert back into uint8, and make original image
   center = np.uint8(center)
   res = center[label.flatten()]
   res2 = res.reshape((img.shape))
   return res2

# feature extraction and do the quantization
def FeatureExtractorGrayscale(img, size, column):
        feature = []
        img = cv2.imread(data_path + globals()[data_type[i]][j][0], cv2.IMREAD_COLOR)
        img = cv2.resize(img, (size, size))
        
        Z = img.reshape((-1,3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
        # res1 = colorQuant(img, Z, 2, criteria)
        res2 = colorQuant(img, Z, 2, criteria)
        # res3 = colorQuant(img, Z, 8, criteria)
        
        # plt.subplot(221),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        
        # plt.subplot(222),plt.imshow(cv2.cvtColor(res1, cv2.COLOR_BGR2RGB))
        # plt.title('K=2'), plt.xticks([]), plt.yticks([])
        
        # plt.subplot(223),plt.imshow(cv2.cvtColor(res2, cv2.COLOR_BGR2RGB))
        # plt.title('K=4'), plt.xticks([]), plt.yticks([])
        
        # plt.subplot(224),plt.imshow(cv2.cvtColor(res3, cv2.COLOR_BGR2RGB))
        # plt.title('K=8'), plt.xticks([]), plt.yticks([])
        # plt.show()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray.reshape(column, size, size)
        # Normalize feature vector
        gray = cv2.normalize(gray, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        return gray

# Function for caculating TOP-1, TOP-5 score of prediction
def top1_5_score(test_labels, prediction, model):
    # label = model.classes
    label = np.arange(0, 50, 1)
    top1_score = 0
    top5_score = 0
    test_labels = np.argmax(test_labels, axis = -1)
    for i in tqdm(range(len(test_labels))):
        top5_ans = np.argpartition(prediction[i], -5)[-5:]
        if int(test_labels[i]) in label[top5_ans]:
            top5_score = top5_score + 1
        if int(test_labels[i]) == label[np.argmax(prediction[i])]:
            top1_score = top1_score + 1
    # print(top1_score/len(test_labels) , top5_score/len(test_labels))
    return top1_score/len(test_labels) , top5_score/len(test_labels) 

# Split images and labels
for i in range(len(data_type)):
    for j in tqdm(range(len(globals()[data_type[i]]))):
        img = FeatureExtractorGrayscale(globals()[data_type[i]][j][0], size, column)
        label = globals()[data_type[i]][j][1]
        globals()[data_type[i] + '_images'].append(img)
        globals()[data_type[i] + '_labels'].append(int(label)) 
    globals()[data_type[i] + '_images'] = np.array(globals()[data_type[i] + '_images']).reshape(len(globals()[data_type[i] + '_images']), column, size, size)
    globals()[data_type[i] + '_labels'] = np.eye(len(np.unique(globals()[data_type[i] + '_labels'])), dtype = np.int16)[globals()[data_type[i] + '_labels']]
    globals()[data_type[i] + '_images'] = globals()[data_type[i] + '_images'].reshape(globals()[data_type[i] + '_images'].shape[0], size, size, column)

# model build
model = Sequential()
#Layer 1
#Conv Layer 1
model.add(Conv2D(filters = 6, 
                 kernel_size = 5, 
                 strides = 1, 
                 activation = 'relu', 
                 input_shape = (size, size, column)))
#Pooling layer 1
model.add(MaxPooling2D(pool_size = 2, strides = 2))
#Layer 2
#Conv Layer 2
model.add(Conv2D(filters = 16, 
                 kernel_size = 5,
                 strides = 1,
                 activation = 'relu',
                 input_shape = (30, 30, 6)))
#Pooling Layer 2
model.add(MaxPooling2D(pool_size = 2, strides = 2))
#Flatten
model.add(Flatten())
#Layer 3
#Fully connected layer 1
model.add(Dense(units = 9720, activation = 'relu'))
#Layer 4
#Fully connected layer 2
model.add(Dense(units = 480, activation = 'relu'))
#Layer 5
#Output Layer
model.add(Dense(units = 50, activation = 'softmax'))
 
optimizer = keras.optimizers.Adam(learning_rate = 0.0001) 

model.compile(loss = 'categorical_crossentropy', optimizer = optimizer) # loss = 'categorical_crossentropy', metrics = ['accuracy']

callback = EarlyStopping(monitor = 'val_loss', patience = 5)
# Reduce = ReduceLROnPlateau(monitor = 'val_loss',
#                            factor = 0.5,
#                            patience = 1,
#                            model = 'auto',
#                            min_lr = 0)

start_time = time.time()
history = model.fit(train_images, train_labels, validation_data = (val_images, val_labels), batch_size = 128, epochs = 10, callbacks = [callback])
print("--- %s seconds ---" % (time.time() - start_time))

pred_labels = model.predict(test_images)
results = top1_5_score(test_labels, pred_labels, history)

# plot Loss 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
    








