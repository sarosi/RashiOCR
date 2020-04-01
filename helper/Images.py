#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 17:10:31 2020

@author: GaborSarosi
"""

def isbw(img):
    #img is a numpy.ndarray, loaded using cv2.imread
    if len(img.shape) > 2:
        looks_like_rgbbw = not False in ((img[:,:,0:1] == img[:,:,1:2]) == (img[:,:,1:2] ==  img[:,:,2:3]))
        looks_like_hsvbw = not (True in (img[:,:,0:1] > 0) or True in (img[:,:,1:2] > 0))
        return looks_like_rgbbw or looks_like_hsvbw
    else:
        return True