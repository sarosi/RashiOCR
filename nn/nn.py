#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 01:27:41 2020

@author: GaborSarosi
"""

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from helper import Gimatrias as gm
import csv
from datetime import datetime


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

#<><><><> MODEL <><><><>
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1003)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=30)

#<><><><> TEST SET <><><><>
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

#<><><><> PREDICTIONS <><><><>
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

#<><><><> FINDING THE WRONG PREDICTIONS <><><><>
def evaluate_mistakes(save_to_csv: False, print_to_screen: True):
#    num_mistakes = 0
#    for idx, pred in enumerate(predictions):
#        if not np.argmax(predictions[idx]) == test_labels[idx]:
#            print(idx, 'test:', gm.full_name_of(test_labels[idx]), 'prediction:', gm.full_name_of(np.argmax(predictions[idx])))
#            num_mistakes = num_mistakes+1
#    print('Number of mistakes = ', num_mistakes)
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    if save_to_csv:
        csvfile = csv.writer(open('evaluate/test_mistakes'+dt_string+'.csv', 'w', newline=''))
        #fieldnames = ['Reality', 'Prediction']
        #writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #csvfile.writeheader()
    num_mistakes = 0
    for idx, pred in enumerate(predictions):
        if not np.argmax(predictions[idx]) == test_labels[idx]:
            test_example = gm.full_name_of(test_labels[idx])
            predicted = gm.full_name_of(np.argmax(predictions[idx]))
            if print_to_screen:
                print(idx, 'test:', test_example, 'prediction:', predicted)
            if save_to_csv:
                row = [test_example, predicted]
                csvfile.writerow(row)
            num_mistakes = num_mistakes+1
    print('Number of mistakes = ', num_mistakes)

evaluate_mistakes(True, True)

