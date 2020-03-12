#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:17:39 2020

@author: GaborSarosi
"""

import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')

import cv2
from matplotlib import pyplot as plt
import os
import helper


def clean_image(filename, threshold):
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if not helper.isbw(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("changed to bw")
    #print(image)
    plt.imshow(image)
    plt.show()
    
    height, width = image.shape
    for x in range(0,height):
        for y in range(0,width):
            pixel = image[x,y]
            if pixel < threshold:
                image[x,y] = max(0, image[x,y]-90)  
            if pixel > threshold:
                image[x,y] = 255
    print(image)
    plt.imshow(image)
    plt.show()
    cv2.imwrite(filename, image)
    #print(image)
    

for filename in os.listdir('extracted_letters/workbench'):
    if filename == '.DS_Store':
        continue
    print(filename)
    clean_image(filename, 150)
    print('-cleaned')
 
    
clean_image('extracted_letters/workbench/test_0297.jpg', 190)
    