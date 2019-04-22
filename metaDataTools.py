#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 13:47:08 2019

@author: lukas
"""
import json,config,os

    
#very dangerous!!!
def generateFiles():
    if not os.path.exists('metaData'):
        os.makedirs('metaData')
    for ident in range(1000):
        file_name = 'metaData/labels'+str(ident%1000).zfill(4)+'.json'
        with open(file_name,'w+') as f:
            f.write('[')
            f.write(']')
        

def readFromJson():
    possibleLabels = {}
    for label in config.labels:
        possibleLabels[label['label']] = 0
    numImages = 0
    for ident in range(1000):
        print(ident)
        file_name = 'metaData/labels'+str(ident%1000).zfill(4)+'.json'
        with open(file_name,'r') as f:
            json_file = [line.rstrip('\n') for line in f]
            json_file = ''.join(json_file)
            json_file = json.loads(json_file)
            for img in json_file:
                numImages += 1
                labels = img['labels']
                for label in labels:
                    possibleLabels[label['label']] += 1
                ident = img['id']
    return (numImages, possibleLabels)

def printJson():
    import time
    time1 = time.time()
    print(readFromJson())
    time2 = time.time()
    print('readFromJson function took {:.3f} ms'.format((time2-time1)*1000.0))
                
                
import os
def getPath(identifier,rootDir = config.rootDir):
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

def convertToYOLO():
    i = 0
    possibleLabels = {}
    for label in config.labels:
        possibleLabels[label['label']] = str(i)
        i += 1
    print("The labels are:",possibleLabels)
    num = 0
    with open('train.txt','w+') as export:
        for ident in range(1000):
            file_name = 'metaData/labels'+str(ident%1000).zfill(4)+'.json'
            with open(file_name,'r') as f:
                json_file = [line.rstrip('\n') for line in f]
                json_file = ''.join(json_file)
                json_file = json.loads(json_file)
                for img in json_file:
                    num += 1
                    string = getPath(img['id'])+' '
                    labels = img['labels']
                    for label in labels:
                        for c in label['box']:
                            string += str(c)+','
                        string += possibleLabels[label['label']]+' '
                    export.write(string+'\n')
                    ident = img['id']
    print("Converted",num,"images.")

def removeDoublesFromJson(data):
    for img in data:
        labels = img['labels']
        print(labels)
        ident = img['id']
        file_name = 'metaData/labels'+str(ident%1000).zfill(4)+'.json'
        with open(file_name,'w+') as f:
            json_file={'id':ident,'labels':labels}
            f.write('[')
            f.write(json.dumps(json_file))#, cls=MyEncoder))
            f.write(']')
            
            
            
            
            
            
            
            
            
            
            
            
            
            