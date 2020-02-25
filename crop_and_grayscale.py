#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:43:43 2020

@author: GaborSarosi
"""

import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')

import cv2
from matplotlib import pyplot as plt
import os

def isbw(img):
    #img is a numpy.ndarray, loaded using cv2.imread
    if len(img.shape) > 2:
        looks_like_rgbbw = not False in ((img[:,:,0:1] == img[:,:,1:2]) == (img[:,:,1:2] ==  img[:,:,2:3]))
        looks_like_hsvbw = not (True in (img[:,:,0:1] > 0) or True in (img[:,:,1:2] > 0))
        return looks_like_rgbbw or looks_like_hsvbw
    else:
        return True

def crop_and_gr_img(img, save_to_file=False, filename=None):
    
    print('Original dimensions: ', img.shape)

    #grayscale (if needed) and crop image
    if not isbw(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cropped = gray[4:28, 4:28]
    else:
        cropped = img[4:28, 4:28]                
        
    print('Cropped dimensions: ', cropped.shape) 
    plt.imshow(cropped)
    plt.show()
    
    #save image to file
    if save_to_file:
        cv2.imwrite(filename, cropped)
        print(filename)
            
path = 'extracted_letters/resized_all'
folder_to_save = path + '/croppedgr/'
for idx, fname in enumerate(os.listdir(path)):
    image = cv2.imread(path + '/' + fname, cv2.IMREAD_UNCHANGED)
    if not image is None:
        filename_to_save = folder_to_save+ str(idx)+ '.jpg'
        print(fname)
        crop_and_gr_img(image, save_to_file=True, filename=filename_to_save)


