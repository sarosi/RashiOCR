#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 00:22:25 2020

@author: GaborSarosi
"""

import pandas as pd


df = pd.read_csv('dataset/train/train.csv')
df_grouped = df['Label'].value_counts()
print(df_grouped)

print(df.tail(2))

