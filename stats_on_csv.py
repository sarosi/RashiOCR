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
    for idx, row in enumerate(rows_of_letter.iterrows()):
        print(idx+1, '/', len(rows_of_letter))
        print(row[1][1])
        if '.jpg' not in row[1][1]:
            continue
        image = cv2.imread('dataset/train/img/' + row[1][1], cv2.IMREAD_GRAYSCALE)
        plt.imshow(image)
        plt.show()


#show_images_of_letter('aleph')
show_stats()

#1747: tet -> mem_sofit
#18, 178, 303, 360, 408, 429, 431, 571, 712, 865 -> samech
#4, 80, 119, 182, 193, 199, 228, 243, 308, 349, 394, 404, 407, 444, 509, 519, 534, 540,
#614, 678, 730, 771, 844, 847, 849, 874, 971, 2021, 2069 -> mem_sofit 
