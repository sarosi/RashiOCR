#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 01:27:41 2020

@author: GaborSarosi
"""

import tensorflow as tf
from tensorflow import keras
import pandas as pd


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