#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 22:37:41 2020

@author: GaborSarosi
"""

import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')

import cv2
from matplotlib import pyplot as plt
import os
from os import path
import re
import csv
import helper

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def full_name_of(character):
    switcher = {
            1: "aleph",
            2: "beth",
            3: "gimel",
            4: "daleth",
            5: "hey",
            6: "vav",
            7: "zayin",
            8: "chet",
            9: "tet",
            10: "yud",
            20: "chaf",
            30: "lamed",
            40: "mem",
            50: "nun",
            60: "samech",
            70: "ayin",
            80: "pe",
            90: "tzadi",
            100: "quf",
            200: "resh",
            300: "shin",
            400: "tav",
            500: "chaf_sofit",
            600: "mem_sofit",
            700: "nun_sofit",
            800: "pe_sofit",
            900: "tzadi_sofit",
            1001: "pas",
            1002: "paspas"
    }
    return switcher.get(character, "no such letter")
    

def name_it():
    lbl = int(input('Name the letter! '))
    char_name = full_name_of(lbl)
    print(char_name)
    return char_name


path_to_images = 'dataset/train/img'
csvname = 'dataset/train/train.csv'

def label_them(path_to_images, csvname, start_index):
    with open(csvname, 'a', newline='') as csvfile: 
        for idx, fname in enumerate(sorted_alphanumeric(os.listdir(path_to_images))):
            if idx < start_index:
                continue
            if fname == '.DS_Store':
                continue
            fieldnames = ['Label', 'Filename']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            #writer.writeheader()
    
            print(fname)
            image = cv2.imread(path_to_images + '/' + fname, cv2.IMREAD_GRAYSCALE)
            if not helper.isbw(image):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            plt.imshow(image)
            plt.show()
            
            char_name = name_it()
            if input('Hit ENTER if sure!'):
                break
            else:
                writer.writerow({'Label': char_name, 'Filename': fname})
                print(char_name + ' is saved for ' + fname)
                
            #TODO: fix the confirmation bug
        
label_them('dataset/train/img', 'dataset/train/train.csv', 1301)
    
    