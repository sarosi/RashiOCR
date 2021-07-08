#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 00:22:25 2020

@author: GaborSarosi
"""

import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')

import cv2
import pandas as pd
from matplotlib import pyplot as plt


def load_dataset(dataset):
    if (dataset == 'train'):
        df = pd.read_csv('dataset/train/train.csv')
        print("training set")
    elif (dataset == 'test'):
        df = pd.read_csv('dataset/test/test.csv')
        print("test set")
    return df
        
    
def dataset_stats(dataset, show):
    df = load_dataset(dataset)
    d_top = df.head()
    if show:
        print(d_top)
    df_grouped_nominal = df['Label'].value_counts()
    df_grouped_percent = df['Label'].value_counts(normalize=True).mul(100).round(1)
    if show:
        print(pd.concat([df_grouped_nominal, df_grouped_percent], axis=1, keys=['counts', '%']))
        print(df.tail(2))
    return df_grouped_nominal, df_grouped_percent

def show_images_of_letter(dataset,letter):
    df = load_dataset(dataset)
    rows_of_letter = df.loc[df['Label'] == letter]
    for idx, row in enumerate(rows_of_letter.iterrows()):
        print(idx+1, '/', len(rows_of_letter))
        print(row[1][1])
        if '.jpg' not in row[1][1]:
            continue
        #TODO: show image from the dataset, not the .jpg!
        image = cv2.imread('dataset/train/img/' + row[1][1], cv2.IMREAD_GRAYSCALE)
        plt.imshow(image)
        plt.show()


#show_images_of_letter('train', 'aleph')
tst_nom, tst_per = dataset_stats('test', False)
trn_nom, trn_per = dataset_stats('train', False)


#18, 178, 303, 360, 408, 429, 431, 571, 712, 865 -> samech
#4, 80, 119, 182, 193, 199, 228, 243, 308, 349, 394, 404, 407, 444, 509, 519, 534, 540,
#614, 678, 730, 771, 844, 847, 849, 874, 971, 2021, 2069 -> mem_sofit 
