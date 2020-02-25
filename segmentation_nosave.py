#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:22:36 2020

@author: GaborSarosi
"""

import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

img_name = "IMG_7610"
img_ext = "jpg"
img = cv2.imread("images/" + img_name + "." + img_ext)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_img = cv2.GaussianBlur(gray_img, (5,5), 0)
ret, im_th = cv2.threshold(gray_img, 90, 255, cv2.THRESH_BINARY_INV)
im_th = cv2.adaptiveThreshold(gray_img, 255, 
cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,75, 10)
im_th = cv2.bitwise_not(im_th)

plt.imshow(img)
plt.show()

ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, 
cv2.CHAIN_APPROX_SIMPLE)
#img_contours = img
#cv2.drawContours(img_contours, ctrs, -1, (0,255,0), 3)

#plt.imshow(img_contours)
#plt.show()
def area_of(rect):
    return rect[2] * rect[3]

def join_rects_of(rect1, rect2):
    x1 = min(rect1[0], rect2[0])
    y1 = min(rect1[1], rect2[1])
    x2 = max((rect1[0] + rect1[2]), (rect2[0] + rect2[2]))
    y2 = max((rect1[1] + rect1[3]), (rect2[1] + rect2[3]))
    height = x2 - x1
    width = y2 - y1
    return x1,y1,height,width

def iou_of(rect1, rect2):
    ix1 = max(rect1[0], rect2[0])
    iy1 = max(rect1[1], rect2[1])
    ix2 = min((rect1[0] + rect1[2]), (rect2[0] + rect2[2]))
    iy2 = min((rect1[1] + rect1[3]), (rect2[1] + rect2[3]))
         
    if (ix2 - ix1 < 0) or (iy2 - iy1 < 0):
        return 0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = area_of(rect1) + area_of(rect2) - intersection
    print("   union=", union)
    print("   intersection=", intersection)
    return intersection / union

def get_contour_precedence(contour, cols):
    origin = cv2.boundingRect(contour)
    return origin[1] * cols + origin[0]

sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
#ctrs.sort(key=lambda x:get_contour_precedence(x, img.shape[1]))

rects = [cv2.boundingRect(ctr) for ctr in sorted_ctrs]
print("Rects before IOU = ", len(rects))
        
rois = []
next_rect = cv2.boundingRect(sorted_ctrs[-1])
joined_rect = False
last_rect_was_joined = False

for idx,rect in enumerate(rects):
    #filter out too small (probably noise, or dot) or too big (probably more than one letters) 
    if area_of(rect) < 600 or area_of(rect) > 12000:
        continue
    print("->", idx, "---", rect)
    #intersection over union (needed because of the small part of quf and hey)
    iou = iou_of(rect, next_rect)
    if iou >= 0.01 and iou <= 0.8:
        x,y,h,w = join_rects_of(rect, next_rect)
        print("   joined rect_final=", x,y,h,w)
        joined_rect = True
    else:
        x,y,h,w = rect[:]
        joined_rect = False
        
    #wider ones are suspicious, rashi letters are taller than wider
    if h > w:
        continue

    if not last_rect_was_joined:
        cv2.rectangle(img,(x, y), (x + h, y + w), (0,255,0), 2)
        cv2.putText(img, "{}".format(len(rois)), (x,y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,155,100), 2)
    
        length = int(h * 1.6)
        roi = img[y-2:y+w, x-2:x+h]
        rois.append(roi)
        if roi.size > 0 and roi.size <= 40:
            plt.imshow(roi)
            plt.show()
    
    if idx < len(rects)-2:
        next_rect = rects[idx+2]
    last_rect_was_joined = joined_rect
        
print("Rois=", len(rois))
filename = "imageswithrois/" + img_name + "_rois.jpg"
cv2.imwrite(filename, img)
    
plt.imshow(img)
plt.show()



    





