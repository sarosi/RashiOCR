#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:35:10 2020

@author: GaborSarosi
"""

import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')

import cv2
from matplotlib import pyplot as plt
import os


def resize_img(img, max_height=24, box_side=32, save_to_file=False, filename=None):
    
    print('Original Dimensions : ',img.shape)
     
    plt.imshow(img)
    plt.show()
    
    height = img.shape[0]
    width = img.shape[1]
    
    if height >= width:
        height = max_height
        scale_ratio = max_height/img.shape[0]
        width = int(img.shape[1] * scale_ratio)
    else:
        width = max_height
        scale_ratio = max_height/img.shape[1]
        height = int(img.shape[0] * scale_ratio)
    
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    print('Resized Dimensions : ',resized.shape) 
    plt.imshow(resized)
    plt.show()
    
    #normalize image
    #normalized = np.zeros((80, 80))
    #normalized = cv2.normalize(resized, normalized, 0, 255, cv2.NORM_MINMAX)
    
    #print('Normalized Dimensions: ', normalized.shape)
    #plt.imshow(normalized)
    #plt.show()
    
    #add padding to image, center them in 32*32
    if height == width:
        t=b=l=r = 4
    else:
        t=b = int((box_side-height)/2)
        l=r = int((box_side-width)/2)
        if height+2*t == box_side-1:
            b = b+1
        if width+2*l == box_side-1:
            r = r+1
        
    resized_with_padding = cv2.copyMakeBorder(resized.copy(),t,b,l,r, cv2.BORDER_CONSTANT,value=[255,255,255])
    print('Resized with padding Dimensions:', resized_with_padding.shape )
    plt.imshow(resized_with_padding)
    plt.show()
    
    #save image to file
    if save_to_file:
        cv2.imwrite(filename, resized_with_padding)
            
path = 'letters_IMG_7569'

folder_to_save = path + '/resized/'
for idx, fname in enumerate(os.listdir(path)):
    image = cv2.imread(path + '/' + fname, cv2.IMREAD_UNCHANGED)
    if not image is None:
        filename_to_save = folder_to_save+str(idx)+'.jpg'
        resize_img(image, max_height=24, box_side=32, save_to_file=True, filename=filename_to_save)


