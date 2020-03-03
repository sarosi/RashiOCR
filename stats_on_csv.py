#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 00:22:25 2020

@author: GaborSarosi
"""

import pandas as pd


df = pd.read_csv('dataset/train/train.csv')
df_grouped_nominal = df['Label'].value_counts()
df_grouped_percent = df['Label'].value_counts(normalize=True).mul(100).round(1)
print(pd.concat([df_grouped_nominal, df_grouped_percent], axis=1, keys=['counts', '%']))
#print(df_grouped)

print(df.tail(2))

