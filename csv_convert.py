#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 01:42:08 2020

@author: GaborSarosi
"""

import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')

import cv2
import pandas as pd
import csv
from helper import Gimatrias as gm


columnNames = list()

csv_train_fnames = 'dataset/train/train.csv'
csv_train_pixels = 'dataset/train/train_flat_gim.csv'

csv_test_fnames = 'dataset/test/test.csv'
csv_test_pixels = 'dataset/test/test_flat_gim.csv'

#!!!!! change according to train or test 
#!!!!! and in csvfile = csv.wrtiter...
#!!!!! and in image =  cv2.imread...
df = pd.read_csv(csv_train_fnames)

csvfile = csv.writer(open(csv_train_pixels, 'a', newline=''))
for idx, fname in enumerate(df['Filename'].tolist()):
    image = cv2.imread('dataset/train/img/'+fname, cv2.IMREAD_GRAYSCALE)
    image_array = []
    height, width = image.shape
    for x in range(0,height):
        for y in range(0,width):
            pixel_value = image[x,y]
            image_array.append(pixel_value)
    label = df.at[idx, 'Label']
    image_array.insert(0, gm.gimatria_of(label))
    csvfile.writerow(image_array)
        
    










