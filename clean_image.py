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
import Images as im


def clean_image(filename, threshold):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if not im.isbw(image):
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
    

for filename in os.listdir('images/screenshots/resized'):
    if filename == '.DS_Store':
        continue
    print(filename)
    clean_image(filename, 150)
    print('-cleaned')

    
#clean_image('extracted_letters/workbench/test_0316.jpg', 190)
    
#clean_image('images/screenshots/resized/resized13.jpg', 190)
