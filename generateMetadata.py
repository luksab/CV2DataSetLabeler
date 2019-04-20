#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 17:44:43 2019

@author: lukas
"""

import json, os, numpy as np
import pickle
import config
from timeit import default_timer as timer
def getData(i):
    data = []
    for line in open(config.origMeta+str(i).zfill(2),'r'):
        data.append(json.loads(line))
    return data

def exsits(identifier,rootDir = config.rootDir):
    string = rootDir
    string += '0'+str(int(identifier)%1000).zfill(3)+'/'
    string += str(int(identifier))+'.'
    if(os.path.isfile(string+'jpg')):
        string += 'jpg'
    elif(os.path.isfile(string+'png')):
        string += 'png'
    else:
        return False
    return True

print("Generating metadata for",config.origMetaNumPics,"images.")
start = timer()
erg = np.zeros((config.origMetaNumPics,5))
j = 0
for i in range(config.origMetaNumFiles):
    data = getData(i)
    print("Generating metadata for file",i+1,"out of",config.origMetaNumFiles)
    for dat in data:
        if exsits(dat['id']):
            f = 2 if dat['rating']=='e' else 0
            erg[j] = np.array([int(dat['id']),int(dat['image_width']),int(dat['image_height']),1 if dat['rating']=='q' else f,
               int(dat['score'])])
            j += 1
erg = erg[erg[:,0] != 0]
pickle.dump( np.int32(erg), open(config.metadata, "wb" ) )

import metaDataTools
metaDataTools.generateFiles()

print("Generating took",timer()-start,"seconds.")
