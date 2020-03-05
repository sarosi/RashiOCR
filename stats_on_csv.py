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

df = pd.read_csv('dataset/train/train.csv')

def show_stats():
    df_grouped_nominal = df['Label'].value_counts()
    df_grouped_percent = df['Label'].value_counts(normalize=True).mul(100).round(1)
    print(pd.concat([df_grouped_nominal, df_grouped_percent], axis=1, keys=['counts', '%']))
    print(df.tail(2))

def show_images_of_letter(letter):
    rows_of_letter = df.loc[df['Label'] == letter]
    #print(rows_of_letter)
    #print(len(rows_of_letter))
    for row in rows_of_letter.iterrows():
        print(row[1][1])
        if '.jpg' not in row[1][1]:
            continue

        image = cv2.imread('dataset/train/img/' + row[1][1], cv2.IMREAD_GRAYSCALE)
        plt.imshow(image)
        plt.show()


#show_images_of_letter('ayin')
show_stats()
