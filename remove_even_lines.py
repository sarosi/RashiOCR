#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:46:09 2020

@author: GaborSarosi
"""

import csv
import string

def print_content():
    csvname = 'dataset/train/train.csv'
    csv_edited_name = 'dataset/train/train_edited.csv'
    with open(csvname, 'r', newline = '') as inp, open(csv_edited_name, 'w', newline = '') as out:
        fieldnames = ['Label', 'Filename']
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(csv.reader(inp)):
            lbl = row[0]
            fname = ''
            if len(row) >= 2:
                fname = row[1]
            print(idx, ': ', lbl)
            if not fname == '':
                print('-', fname)
            
            #writer.writerow({'Label': lbl, 'Filename': fname})
    
    
#with open(csvname, 'r+', newline='') as csvfile:
#    reader = csv.reader(csvfile)
#    for row in reader:
#        print(row[0])
        
        
#fix the scv if some rows are messed up
def fix_this_csv(csv_with_path, new_csv_with_path):
    with open(csv_with_path, 'r', newline = '') as old, open(new_csv_with_path, 'w', newline = '') as new:
        #fieldnames = ['Label', 'Filename']
        new_dict = []
        for idx, row in enumerate(csv.reader(old)):
            if len(row) < 2:
                lbl, fname = row[0].split(";")
            else:
                lbl, fname = row
                if fname[-1] == ';':
                    fname = fname[0:-1].strip()
            print(idx, ': ', lbl, ' ', fname)
            new_dict.append([lbl, fname])
            #print(new_dict)
        writer = csv.DictWriter(new, fieldnames = new_dict[0])
        writer.writeheader()
        for idx, obj in enumerate(new_dict):
            if idx == 0:
                continue
            writer.writerow({'Label': obj[0], 'Filename': obj[1]})
        
            
    
fix_this_csv('dataset/train/train.csv', 'dataset/train/train_fixed.csv')