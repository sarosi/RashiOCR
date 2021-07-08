#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 01:27:41 2020

@author: GaborSarosi
"""

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPool2D
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from helper import Gimatrias as gm
from datetime import datetime
import os
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator



train_df = pd.read_csv(f'dataset/train/train_flat_gim.csv', header=None)
test_df = pd.read_csv(f'dataset/test/test_flat_gim.csv', header=None)

train_labels = train_df.loc[:,0].values
train_images = train_df.loc[:,1:576].values

test_labels = test_df.loc[:,0].values
test_images = test_df.loc[:,1:576].values

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
#model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(24,24,1)))
#model.add(Conv2D(64, kernel_size=3, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(128, kernel_size=3, activation='relu'))
#model.add(Conv2D(128, kernel_size=3, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Flatten())
#model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(1003, activation="softmax"))


model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu', input_shape=(24,24,1)))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1003, activation='softmax'))

epochs = 30
batch_size = 64

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

model.summary()

image_gen=ImageDataGenerator(rotation_range=10,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.1,zoom_range=0.1,horizontal_flip=False,vertical_flip=False,fill_mode='nearest')

model.fit_generator(image_gen.flow(train_images, train_labels, batch_size=batch_size), epochs=epochs, validation_data = (test_images, test_labels), callbacks = [learning_rate_reduction])


#model.fit(train_images, train_labels, epochs=30)

#<><><><> TEST SET <><><><>
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest loss:', test_loss)
print('Test accuracy:', test_acc, '\n')

#<><><><> PREDICTIONS <><><><>
probability_model = keras.Sequential([model, keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

now = datetime.now()
dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")


#<><><><> FINDING THE WRONG PREDICTIONS <><><><>   
def evaluate_mistakes(save_to_csv: False, print_to_screen: True):
#    #if save_to_csv:
#        csvfile = csv.writer(open('evaluate/test_mistakes'+dt_string+'.csv', 'w', newline=''))
#        fieldnames = ['Reality', 'Prediction']
#        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#        csvfile.writeheader()
    print("\n ---- WRONG PREDICTIONS ---- ")
    num_mistakes = 0 
    for idx, pred in enumerate(predictions):
        label_at_idx = np.argmax(test_labels[idx])
        if not np.argmax(predictions[idx]) == label_at_idx:
            test_example = gm.full_name_of(label_at_idx)
            predicted = gm.full_name_of(np.argmax(predictions[idx]))
            if print_to_screen:
                print(idx, 'test:', test_example, 'prediction:', predicted)
#            if save_to_csv:
#                row = [test_example, predicted]
#                csvfile.writerow(row)
            num_mistakes = num_mistakes+1
    print('Number of mistakes = ', num_mistakes, '\n')

#<><><><> SAVING THE MODEL <><><><>
def save_my_weights(num_mistakes):
    modelname = "rashinet" + str(num_mistakes) + "_" + dt_string
    os.chdir("models")
    os.mkdir(modelname)
    os.chdir(modelname)
    
    model_yaml = model.to_yaml()
    with open(modelname+".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    
    model_json = model.to_json()
    with open(modelname+".json", "w") as json_file:
        json_file.write(model_json)
        
    h5name = modelname + '.h5'
    model.save_weights(h5name)
        
    
evaluate_mistakes(True, True)
#save_my_weights(num_mistakes)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph
