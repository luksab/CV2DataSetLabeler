#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:04:37 2019

@author: lukas
"""

import cv2
import numpy as np
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
import pickle
Data = pickle.load( open( '/home/lukas/Documents/Py/Metadata/metadata.pk', "rb" ) )
ids = Data[Data[:,3] >= 1][:,0]

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

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
UIHeight= 40
showLabels = True
hideLabels = False
changedLabels = False

def selectLabel(x,y):
    global labels
    for label in labels:
        print(label['box'],'     ',x,y)
        if(label['box'][0] < x < label['box'][2] and label['box'][1] < y < label['box'][3]):
            if('active' in label):
                del label['active']
                continue
            else:
                label['active'] = True
                print(hasattr(label, 'active'))
                print(label)
            return True
    return False

def deleteSelected():
    global labels,changedLabels
    labels[:] = [d for d in labels if d.get('active') != True]
    changedLabels = True


def predict():
    global labels, img
    labels = m.detect(img,min_prob=cv2.getTrackbarPos('minP','image')/100)

# mouse callback function
def mouseEvent(event,nx,ny,flags,param):
    global ix,iy,x,y,labels,state,UIWidth,select,selectedLabel,possibleLabels,changedLabels

    if(nx < UIWidth+155*showLabels):
        if(event == cv2.EVENT_LBUTTONDOWN):
            Button = int((ny-UIHeight)/UIWidth)
            print(Button)
            if(Button == 0):
                select = not select
            if(Button == 1):
                predict()
                changedLabels = True
            else:
                selectedLabel = Button-2
        return
    if(not state):
        ix,iy = nx-UIWidth-155*showLabels,ny-UIHeight
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if(select):
            if(selectLabel(nx-UIWidth-155*showLabels,ny-UIHeight)):
                pass
            return
        if(state):
            box = [min(ix,x),min(iy,y),max(ix,x),max(iy,y)]
            labels.append({'box': box, 'score': 1, 'label': possibleLabels[selectedLabel]['label']})
            changedLabels = True
        state = not state

    elif event == cv2.EVENT_RBUTTONUP:
        state = not state
    x,y = nx-UIWidth-155*showLabels,ny-UIHeight

img = np.zeros((512,512,3), np.uint8)

cv2.namedWindow('image',5)
cv2.setMouseCallback('image',mouseEvent)
def nothing(x):
    return
cv2.createTrackbar('minP','image',30,100,nothing)
cv2.resizeWindow('image', 1920, 1080)


def doneWithImage(write = True, goBack = False):
    global ix,iy,x,y,labels,state,lastLabels,ident,changedLabels
    if(write and changedLabels):
        removeDoublesFromJson()
        with open('lastID.txt','r+') as f:
            lastID = int(next(f))
            if(ident > lastID):
                f.seek(0)
                f.write(str(ident))
    changedLabels = False
    labels = []
    lastLabels = []
    if(goBack):
        if(ident>1):
            i = ident-1
            while(not (i in ids) or not getPath(i)):
                i -= 1
        else:
            i = 1
    else:
        i = ident+1
        while(not (i in ids) or not getPath(i)):
            i += 1
    readImage(i)
    ident = i
    data = readLabels()
    for image in data:
        if(image['id'] == i):
            labels = image['labels']
    
def jumpToLast():
    with open('lastID.txt') as f:
        lastID = int(next(f))
    global ident
    ident = lastID+1
    doneWithImage(False,True)


def readLabels():
    file_name = 'metaData/labels'+str(ident%1000).zfill(4)+'.json'
    with open(file_name) as f:
        json_file = [line.rstrip('\n') for line in f]
        json_file = ''.join(json_file)
        #print(json_file)
        return json.loads(json_file)
    
def removeDoublesFromJson():
    file_name = 'metaData/labels'+str(ident%1000).zfill(4)+'.json'
    with open(file_name,'r+') as f:
        json_file = [line for line in f]
        json_file = ''.join(json_file)
        json_file = json.loads(json_file)
        if(len(labels)>0):
            json_file.append({'id':ident,'labels':labels})
        json_file.reverse()#reverse to keep newest
        
        seen_titles = set()
        noDuplicates = []
        for obj in json_file:
            if obj['id'] not in seen_titles:
                noDuplicates.append(obj)
                seen_titles.add(obj['id'])
        f.seek(0)
        noDuplicates.reverse()#reverse back, just because
        f.write(json.dumps(noDuplicates))#, cls=MyEncoder))
    
    
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


height, width, channels = 0,0,0
def readImage(idPic):
    global ident, img, height, width, channels
    ident = idPic
    img = cv2.imread(getPath(ident))
    height, width, channels = img.shape
    #cv2.resizeWindow('image', width, height)

possibleLabels = [{'label':'label1','color': (247, 206, 118)}, {'label':'label2', 'color': (66, 244, 137)}}]
lastLabels = []
def find_label(label_name):
    global possibleLabels
    for label in possibleLabels:
        if label['label'] == label_name:
            return label
    return False

selectedLabel = 0
def makeUIV(height):
    UI = np.zeros((height,UIWidth,3), np.uint8)
    UI[:] = (128,128,128)
    
    #Select Box
    cv2.circle(UI,(int(UIWidth/2),int(UIWidth/3)+5),int(UIWidth/3),(255,255,255),5,cv2.LINE_AA)
    if(select):
        cv2.circle(UI,(int(UIWidth/2),int(UIWidth/3)+5),int(UIWidth/3),(255,255,255),-1,cv2.LINE_AA)
    #Predict
    cv2.circle(UI,(int(UIWidth/2),int(UIWidth/3)+5+UIWidth),int(UIWidth/3),(0,255,255),5,cv2.LINE_AA)
        
    for i, label in enumerate(possibleLabels,start = 2):
        cv2.circle(UI,(int(UIWidth/2),int(UIWidth/3)+5+i*UIWidth),int(UIWidth/3),label['color'],5,cv2.LINE_AA)
        if(selectedLabel == i-2):
            cv2.circle(UI,(int(UIWidth/2),int(UIWidth/3)+5+i*UIWidth),int(UIWidth/3),label['color'],-1,cv2.LINE_AA)
    
    if(showLabels):
        Labels = np.zeros((height,155,3), np.uint8)
        Labels[:] = (128,128,128)
        for i, label in enumerate(possibleLabels,start = 2):
            TpLeft = (0,int(UIWidth/3)+13+i*UIWidth)
            #cv2.putText(Labels, label['label'], TpLeft, font, 0.75, (0,), 4,cv2.LINE_AA)
            cv2.putText(Labels, label['label'], TpLeft, font, 0.75, (255,), 2,cv2.LINE_AA)
        UI = np.hstack((Labels, UI))
    return UI

def makeUIH(width):
    UI = np.zeros((UIHeight,width,3), np.uint8)
    UI[:] = (128,128,128)
    cv2.putText(UI, str(ident), (int(width/2),30), font, 0.75, (255,), 2,cv2.LINE_AA)
    return UI

doneWithImage()

font = cv2.FONT_HERSHEY_SIMPLEX
sinceLast = 10
while(1):
    temp = img.copy()
    if(state):
        cv2.rectangle(temp,(ix,iy),(x,y),possibleLabels[selectedLabel]['color'],1)
    if(not hideLabels):
        for box in labels:
            overlay = temp.copy()
            TpLeft = (box['box'][0], box['box'][1])
            BRight = (box['box'][2], box['box'][3])
            if('active' in box):
                color = find_label(box['label'])['color']
                color = (color[0]+100,color[1]+100,color[2]+100)
                cv2.rectangle(temp, TpLeft, BRight, color, 2)
            else:
                cv2.rectangle(overlay, TpLeft, BRight, find_label(box['label'])['color'], -1)
                cv2.addWeighted(overlay, 0.5, temp, 1-0.5,0, temp)
                cv2.rectangle(temp, TpLeft, BRight, find_label(box['label'])['color'], 1)
            cv2.putText(temp, box['label'], TpLeft, font, 0.75, (0,), 4,cv2.LINE_AA)
            cv2.putText(temp, box['label'], TpLeft, font, 0.75, (255,), 2,cv2.LINE_AA)
    
    #draw mouseLines
    if(select):
        color = (100,100,100)
    else:
        color = (200,200,200)
    cv2.line(temp,(0,y),(width,y),color)
    cv2.line(temp,(x,0),(x,height),color)
    
    
    temp = np.hstack((makeUIV(height), temp))
    temp = np.vstack((makeUIH(width+UIWidth+155*showLabels), temp))
    cv2.imshow('image',temp)
    k = cv2.waitKey(15) & 0xFF
    #print(k,sinceLast)
    if(sinceLast > 15):
        if k == ord(' '):
            doneWithImage()
        elif k == 99:#Right/c
            doneWithImage(False)
        elif k == 120:#Left/x
            doneWithImage(False,True)
        elif k == 115:#s
            if(select):
                for box in labels:
                    if('active' in box):
                        del box['active']
            select = not select
        elif k == 101:#e
            deleteSelected()
        elif k == 106:#j
            jumpToLast()
        elif k == 112:#p
            predict()
        elif k == 27:#esc
            break
        elif k == 104:#h
            hideLabels = not hideLabels
        elif k == 122:#z
            if(len(labels)>0):
                lastLabels.append(labels.pop())
                changedLabels = True
        elif k == 121:#y
            if(len(lastLabels)>0):
                labels.append(lastLabels.pop())
                changedLabels = True
        elif(k>=49 and k<49+len(possibleLabels)):
            for box in labels:
                if('active' in box):
                    box['label'] = possibleLabels[k-49]['label']
            selectedLabel = k-49
        elif k != 255:
            print(k)
    if(k != 255):
        sinceLast = 0
    sinceLast += 1
cv2.destroyAllWindows()
