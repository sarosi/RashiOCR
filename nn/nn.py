#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 01:27:41 2020

@author: GaborSarosi
"""

import tensorflow as tf
#from tensorflow import keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from helper import Gimatrias as gm
import csv
from datetime import datetime
import os


train_df = pd.read_csv(f'../dataset/train/train_flat_gim.csv', header=None)
test_df = pd.read_csv(f'../dataset/test/test_flat_gim.csv', header=None)

#print("training set")
#print(train_df.head)
#print("test set")
#print(test_df.head)

train_labels = train_df.loc[:,0].values
#print("train labels")
#print(train_labels)
train_images = train_df.loc[:,1:576].values
#print("train images")
#print(train_images)

test_labels = test_df.loc[:,0].values
#print("test labels")
#print(test_labels)
test_images = test_df.loc[:,1:576].values
#print("test images")
#print(test_images)

train_images = np.expand_dims(train_images, axis=0)
test_images = np.expand_dims(test_images, axis=0)

train_size = train_labels.size

train_images = train_images.reshape(train_labels.size, 24, 24, 1)
test_images = test_images.reshape(test_labels.size, 24, 24, 1)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#needed only because test data does not have paspas, so it is only 1002 long
#TODO: add paspas to test_data!!!!! and then remove this line
test_labels = np.hstack((test_labels, np.zeros((test_labels.shape[0], 1), dtype=test_labels.dtype)))

set_image_dim_ordering="th"

#<><><><> MODEL <><><><>

model = keras.Sequential()
#model.add(keras.layers.InputLayer(()))
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(24,24,1)))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1003, activation="softmax"))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='adam',
#              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#              metrics=['accuracy'])

model.summary()

model.fit(train_images, train_labels, epochs=18)

#<><><><> TEST SET <><><><>
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest loss:', test_loss)
print('\nTest accuracy:', test_acc)

#<><><><> PREDICTIONS <><><><>
probability_model = keras.Sequential([model, keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

now = datetime.now()
dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    
#<><><><> FINDING THE WRONG PREDICTIONS <><><><>
def evaluate_mistakes(save_to_csv: False, print_to_screen: True):
    if save_to_csv:
        csvfile = csv.writer(open('evaluate/test_mistakes'+dt_string+'.csv', 'w', newline=''))
        #fieldnames = ['Reality', 'Prediction']
        #writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #csvfile.writeheader()
    num_mistakes = 0
    for idx, pred in enumerate(predictions):
        label_at_idx = np.argmax(test_labels[idx])
        if not np.argmax(predictions[idx]) == label_at_idx:
            test_example = gm.full_name_of(label_at_idx)
            predicted = gm.full_name_of(np.argmax(predictions[idx]))
            if print_to_screen:
                print(idx, 'test:', test_example, 'prediction:', predicted)
            if save_to_csv:
                row = [test_example, predicted]
                csvfile.writerow(row)
            num_mistakes = num_mistakes+1
    print('Number of mistakes = ', num_mistakes)

#<><><><> SAVING THE MODEL <><><><>
def save_my_weights(num_mistakes):
    modelname = "rashinet" + str(num_mistakes) + "_" + dt_string
    os.chdir("models")
    os.mkdir(modelname)
    os.chdir(modelname)
    
    model_yaml = model.to_yaml
    with open(modelname+".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    
    model_json = model.to_json
    with open(modelname+".json", "w") as json_file:
        json_file.write(model_json)
        
    model.save_weights(modelname)
    
    
evaluate_mistakes(True, True)

