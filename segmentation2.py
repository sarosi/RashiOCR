#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:28:57 2020

@author: GaborSarosi
"""

import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')

import cv2
import numpy as np
from matplotlib import pyplot as plt

#read image
img = cv2.imread('rscript2.jpg')


#grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#binarize 
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

#find contours
ctrs, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
cv2.CHAIN_APPROX_SIMPLE)


#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
print(len(sorted_ctrs))

plt.imshow(img)
plt.show()

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)   

    # Getting ROI
    roi = img[y:y+h, x:x+w]

    # show ROI
    #cv2.imwrite('roi_imgs.png', roi)
    cv2.imshow('charachter'+str(i), roi)
    cv2.rectangle(img,(x,y),( x + w, y + h ),(90,0,255),2)
    print(roi)

plt.imshow(img)
plt.show()