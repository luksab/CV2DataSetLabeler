#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:04:37 2019

@author: lukas
"""

import cv2
import numpy as np
import json, os
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
        
import config, requests     
        
if not os.path.isfile(config.metadata):
    print("Downloading metadata...")
    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
    
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : config.metaDataURL }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : config.metaDataURL, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, config.metadata)    
    
    

import pickle
Data = pickle.load( open( config.metadata, "rb" ) )
#Data = Data[Data[:,3] < 1]
#ids = Data[Data[:,5] == 1][:,0]
ids = Data[Data[:,3] >= 1][:,0]
    
if config.disableUseGPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
if not 'yolo' in globals() and config.predict:
    from yolo import YOLO
    yolo = YOLO()

labels = []
state = False
select = False
ix,iy = -1,-1
x,y = -1,-1
ident = 0
UIWidth = 40
UIHeight= 40
showLabels = config.showLabels
hideLabels = False
changedLabels = False

skipImgs = [7177,7212,7373]

def getPath(identifier,rootDir = config.rootDir):
    if(getImagesFromInternet):
        return identifier not in skipImgs
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
getImagesFromInternet = not os.path.exists(config.rootDir)

import datetime
session = {'ImagesLabeld': 0, 'TimeSpend': 0, 'StartTime': str(datetime.datetime.now())}
from timeit import default_timer as timer
startedSessionAt = timer()

def selectLabel(x,y):
    global labels,instaDelete
    for label in labels:
        if(label['box'][0] < x < label['box'][2] and label['box'][1] < y < label['box'][3]):
            if('active' in label):
                del label['active']
                continue
            else:
                label['active'] = True
            return True
    return False

def deleteSelected():
    global labels,changedLabels
    labels[:] = [d for d in labels if d.get('active') != True]
    changedLabels = True


def predict():
    global labels, img, lastLabels
    if(config.predict):
        lastLabels += labels
        labels = yolo.detect_box_np(img)
    else:
        printError("I can't use the predictor. It is disabled in config.json")

# mouse callback function
def mouseEvent(event,nx,ny,flags,param):
    global ix,iy,x,y,labels,state,UIWidth,select,selectedLabel,possibleLabels,changedLabels, instaDelete

    if(nx < UIWidth+155*showLabels):
        if(event == cv2.EVENT_LBUTTONDOWN):
            Button = int((ny-UIHeight)/UIWidth)
            if(Button == 0):
                select = not select
            if(Button == 1):
                predict()
                changedLabels = True
            elif Button-2 >= 0 and Button-2 <= 9:
                switchSelecting = True
                for box in labels:
                    if('active' in box):
                        box['label'] = possibleLabels[Button-2]['label']
                        switchSelecting = False
                if switchSelecting:
                    for box in labels:
                        if('active' in box):
                            del box['active']
                selectedLabel = Button-2
        return
    if(not state):
        ix,iy = nx-UIWidth-155*showLabels,ny-UIHeight
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if select or instaDelete:
            selectLabel(nx-UIWidth-155*showLabels,ny-UIHeight)
            if instaDelete:
                deleteSelected()
                instaDelete = False
            return
        if(state):
            box = [min(ix,x),min(iy,y),max(ix,x),max(iy,y)]
            labels.append({'box': box, 'score': 1, 'label': possibleLabels[selectedLabel]['label']})
            changedLabels = True
        state = not state

    x,y = nx-UIWidth-155*showLabels,ny-UIHeight

img = np.zeros((512,512,3), np.uint8)

cv2.namedWindow('image',cv2.WINDOW_NORMAL)#5
cv2.setMouseCallback('image',mouseEvent)
cv2.resizeWindow('image', 1920, 1080)


def doneWithImage(write = True, goBack = False):
    global ix,iy,x,y,labels,state,lastLabels,ident,changedLabels
    if(write and changedLabels):
        session['ImagesLabeld'] += 1
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
        i = ident-1
        while not((i in ids) and getPath(i)) and i>1:
            i -= 1
        if i == 0:
            printError('Did not find image')
            i = 1
    else:
        i = ident+1
        while not((i in ids) and getPath(i)):
            i += 1
    print(i)
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
#    data = readLabels()
#    lastID = 0
#    for image in data:
#        if(lastID < image['id']):
#            lastID = image['id']
    ident = lastID+1
    print('jumping to',ident)
    doneWithImage(False,True)


def readLabels():
    file_name = 'metaData/labels'+str(ident%1000).zfill(4)+'.json'
    with open(file_name) as f:
        json_file = [line.rstrip('\n') for line in f]
        json_file = ''.join(json_file)
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
        f.write(json.dumps(noDuplicates, cls=MyEncoder))


import socket
def internet(host = "8.8.8.8", port = 53, timeout = 3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host,port))
        return True
    except Exception as ex:
        print(ex)
        return False


height, width, channels = 0,0,0
def readImage(idPic):
    global ident, img, height, width, channels
    if(getImagesFromInternet):
        if not os.path.isfile(config.saveImgURL+"/"+str(ident)+".png"):
            if(internet()):
                with requests.get(config.url+str(idPic)+".json") as url:
                    if url.status_code == 200:
                        data = json.loads(url.text)
                        if('file_url' in data):
                            with requests.get(data['file_url'], stream=True) as req:
                                req.raw.decode_content = True
                                arr = np.asarray(bytearray(req.content), dtype=np.uint8)
                                img = cv2.imdecode(arr, -1) # 'Load it as it is'
                                if config.saveImgs:
                                    cv2.imwrite(config.saveImgURL+"/"+str(ident)+".png",img)
                                    print('reading img'+config.saveImgURL+"/"+str(ident)+".png")
                                    img = cv2.imread(config.saveImgURL+"/"+str(ident)+".png",cv2.IMREAD_COLOR)
                        else:
                            printError("Didn't find any file in the response from",data['id'])
                            print(data)
                            skipImgs.append(data['id'])
                            doneWithImage(write=False)
                    else:
                        printError("The server didn't respond properly to",url)
                        doneWithImage(write=False)
            else:
                printError("No Internet! - can't download image")
        else:
            print("Found",config.saveImgURL+"/"+str(ident)+".png")
            img = cv2.imread(config.saveImgURL+"/"+str(ident)+".png",cv2.IMREAD_COLOR)
    else:
        ident = idPic
        print(getPath(ident))
        img = cv2.imread(getPath(ident),cv2.IMREAD_COLOR)
    height, width, channels = img.shape
    #cv2.resizeWindow('image', width, height)
    



possibleLabels = config.labels
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

font = cv2.FONT_HERSHEY_SIMPLEX

def printError(errorMsg):
    global error, timeSinceError
    timeSinceError = 0
    error = errorMsg
    print(errorMsg)

def displayError(img,error,center=True, x = None, y = None):
    height = 100
    width = 750
    if(center):
        x = img.shape[1]//2 - width//2
        y = img.shape[0]//2 - height//2
    cv2.rectangle(img, (x,y), (x+width,y+height), (255,255,255), -1)
    cv2.putText(img, error, (x+10,y+height//2), font, 0.75, (0,), 4,cv2.LINE_AA)
    return img

doneWithImage()

timeSinceError = 0
error = ""

sinceLast = 10
lastKey = 0
saveImg = False
instaDelete = False

def main(newIds = ids):
    global selectedLabel, lastKey, saveImg, error, sinceLast, timeSinceError, select, showLabels, hideLabels, changedLabels, ids, instaDelete
    ids = newIds
    print(ids)
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
        
        if(saveImg):
            if not os.path.exists('saved'):
                os.makedirs('saved')
            cv2.imwrite("saved/"+str(ident)+".png",temp)
            saveImg = False
        
        #draw mouseLines
        if instaDelete:
            color = (255,0,0)
        elif select:
            color = (100,100,100)
        else:
            color = (200,200,200)
        cv2.line(temp,(0,y),(width,y),color)
        cv2.line(temp,(x,0),(x,height),color)
        
        
        temp = np.hstack((makeUIV(height), temp))
        temp = np.vstack((makeUIH(width+UIWidth+155*showLabels), temp))
        
        if error != "":
            displayError(temp,error)
        cv2.imshow('image',temp)
        k = cv2.waitKey(15) & 0xFF
        timeSinceError += 1
        if(sinceLast > 20 or lastKey != k):
            if(timeSinceError > 100):
                error = ""
            lastKey = k
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
            elif k == 100:#d
                instaDelete = True
            elif k == 105:#i
                showLabels = not showLabels
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
                switchSelecting = True
                for box in labels:
                    if('active' in box):
                        box['label'] = possibleLabels[k-49]['label']
                        switchSelecting = False
                if switchSelecting:
                    for box in labels:
                        if('active' in box):
                            del box['active']
                selectedLabel = k-49
            elif k == 48:
                switchSelecting = True
                for box in labels:
                    if('active' in box):
                        box['label'] = possibleLabels[10]['label']
                        switchSelecting = False
                if switchSelecting:
                    for box in labels:
                        if('active' in box):
                            del box['active']
                selectedLabel = 9
            elif k == 13:
                saveImg = True
            elif k != 255:
                print(k)
        if(k != 255):
            sinceLast = 0
        sinceLast += 1
    #except:
    #    print('exception')
    if session['ImagesLabeld'] > 0:
        session['TimeSpend'] = int(timer()-startedSessionAt)
        print(json.dumps(session))
        with open('sessions.txt','a+') as f:
            f.write(json.dumps(session)+'\n')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()





















