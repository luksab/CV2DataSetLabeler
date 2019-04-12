#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:04:37 2019

@author: lukas
"""

import cv2
import numpy as np
import random
import detector
import json
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

from keras_retinanet import models
if not 'detection_model' in globals():
    detection_model = models.load_model('detector.h5', backbone_name='resnet101')
m = detector.Detector(detection_model)

labels = []
state = False
select = False
ix,iy = -1,-1
x,y = -1,-1
ident = 0
UIWidth = 40


def selectLabel(x,y):
    global labels
    for label in labels:
        print(label['box'],'     ',x,y)
        if(label['box'][0] < x < label['box'][2] and label['box'][1] < y < label['box'][3]):
            label['active'] = True
            print(label)
            return True
    return False

def deleteSelected():
    global labels
    labels[:] = [d for d in labels if d.get('active') != True]


def predict():
    global labels, img
    labels = m.detect(img,min_prob=cv2.getTrackbarPos('minP','image')/100)
    #print(labels)

# mouse callback function
def mouseEvent(event,nx,ny,flags,param):
    global ix,iy,x,y,labels,state,UIWidth,select

    if(nx < UIWidth):
        if(event == cv2.EVENT_LBUTTONDOWN):
            predict()
        return
    if(not state):
        ix,iy = nx-UIWidth,ny
    if(state):
        x,y = nx-UIWidth,ny
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print(select)
        if(select):
            if(selectLabel(nx-UIWidth,ny)):
                print('selected!')
                select = False
            return
        if(state):
            box = [min(ix,x),min(iy,y),max(ix,x),max(iy,y)]
            labels.append({'box': box, 'score': 1, 'label': 'box?'})
        state = not state
        #print(state)

#    elif event == cv2.EVENT_MOUSEMOVE:
#        
#        x,y = nx,ny

    elif event == cv2.EVENT_LBUTTONUP:
        pass
#        if mode == True:
#            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#        else:
#            cv2.circle(img,(x,y),5,(0,0,255),-1)

img = np.zeros((512,512,3), np.uint8)

cv2.namedWindow('image')
cv2.setMouseCallback('image',mouseEvent)
def nothing(x):
    return
cv2.createTrackbar('minP','image',0,100,nothing)


def doneWithImage(write = True):
    global ix,iy,x,y,labels,state
    if(write):
        pic = {'id':ident,'labels':labels}
        jsonLabels = json.dumps(pic, cls=MyEncoder)
        file = open("guru99.txt","a+")
        file.write(jsonLabels+',\n')
    labels = []
    i = 0
#    while(not getPath(i)):
#        i += int(random.randint(1,100))
    readImage(i)
    
    
    
import os
root = '/run/media/lukas/Data4Tb/danbooru2018/original/'
def getPath(identifier,rootDir = root):
    string = rootDir
    string += '0'+str(int(identifier)%1000).zfill(3)+'/'
    string += str(int(identifier))+'.'
    if(os.path.isfile(string+'jpg')):
        string += 'jpg'
    elif(os.path.isfile(string+'png')):
        string += 'png'
    else:
        return False
    return string

def readImage(idPic):
    global ident, img
    ident = idPic
    print('loading Image')
    img = cv2.imread('/home/lukas/ownCloud/NudeNet-master/desktop-bgs/1855.jpg')#getPath(ident)


def makeUI(height):
    UI = np.zeros((height,UIWidth,3), np.uint8)
    return UI

i = 0
#while(not getPath(i)):
#    i += int(random.randint(1,100))
readImage(i)

font = cv2.FONT_HERSHEY_SIMPLEX
while(1):
    temp = img.copy()
    if(state):
        cv2.rectangle(temp,(ix,iy),(x,y),(0,255,0),2)
    
    for box in labels:
        #cv2.rectangle(temp,(ix,iy),(x,y),(0,255,0),3)
        TpLeft = (box['box'][0], box['box'][1])
        BRight = (box['box'][2], box['box'][3])
        if('active' in box):
            cv2.rectangle(temp, TpLeft, BRight, (255, 255, 255), 2)
        else:
            cv2.rectangle(temp, TpLeft, BRight, (0, 255, 0), 2)
        cv2.putText(temp,box['label'],TpLeft, font, 1,(255,255,255),2,cv2.LINE_AA)
    
    height, width, channels = temp.shape
    temp = np.hstack((makeUI(height), temp))
    cv2.imshow('image',temp)
    k = cv2.waitKey(10) & 0xFF
    if k == ord(' '):
        print('next Image!')
        doneWithImage()
    if k == 8:#Backspace
        print('next Image!')
        doneWithImage(False)
    elif k == 115:#s
        select = not select
        print('select',select)
    elif k == 101:#e
        deleteSelected()
    elif k == 27:#esc
        break
    elif k != 255:
        print(k)

cv2.destroyAllWindows()
























