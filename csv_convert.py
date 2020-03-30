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

def gimatria_of(letter):
    switcher = {
            "aleph": 1,
            "beth": 2,
            "gimel": 3,
            "daleth": 4,
            "hey": 5,
            "vav": 6,
            "zayin": 7,
            "chet": 8,
            "tet": 9,
            "yud": 10,
            "chaf": 20,
            "lamed": 30,
            "mem": 40,
            "nun": 50,
            "samech": 60,
            "ayin": 70,
            "pe": 80,
            "tzadi": 90,
            "quf": 100,
            "resh": 200,
            "shin": 300,
            "tav": 400,
            "chaf_sofit": 500,
            "mem_sofit": 600,
            "nun_sofit": 700,
            "pe_sofit": 800,
            "tzadi_sofit": 900,
            "pas": 1001,
            "paspas": 1002
        }
    return switcher.get(letter, "0")


columnNames = list()

csv_train_fnames = 'dataset/train/train.csv'
csv_train_pixels = 'dataset/train/train_flat_gim.csv'

csv_test_fnames = 'dataset/test/test.csv'
csv_test_pixels = 'dataset/test/test_flat_gim.csv'

#!!!!! change according to train or test 
#!!!!! and in csvfile = csv.wrtiter...
#!!!!! and in image =  cv2.imread...
df = pd.read_csv(csv_test_fnames)

csvfile = csv.writer(open(csv_test_pixels, 'a', newline=''))
for idx, fname in enumerate(df['Filename'].tolist()):
    image = cv2.imread('dataset/test/img/'+fname, cv2.IMREAD_GRAYSCALE)
    image_array = []
    height, width = image.shape
    for x in range(0,height):
        for y in range(0,width):
            pixel_value = image[x,y]
            image_array.append(pixel_value)
    label = df.at[idx, 'Label']
    image_array.insert(0, gimatria_of(label))
    csvfile.writerow(image_array)
        
    










