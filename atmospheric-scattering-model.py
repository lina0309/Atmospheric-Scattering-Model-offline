
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
import tensorflow.contrib.slim as slim
from PIL import Image
import cv2
EPS = 1e-12

image1 = cv2.imread('/home/ouc/data/lina-exp/20180201/testdark/59color/00410_colors.png')
cv2.imshow('image1',image1)
imageb, imageg, imager = cv2.split(image1)
# cv2.imshow('imageb',imageb)
# cv2.imshow('imageg',imageg)
# cv2.imshow('imager',imager)
pred1 = cv2.imread('/home/ouc/data/lina-exp/20180201/testdark/re/00410_colors_55-outputs_b.png')
predb, predg, predr = cv2.split(pred1)
cv2.imshow('pred1',pred1)
cv2.imshow('predb',predb)
cv2.imshow('predg',predg)
cv2.imshow('predr',predr)
depth1 = cv2.imread('/home/ouc/data/lina-exp/20180201/testdark/59depth3/1445_blue-targets.png')
depthb, depthg, depthr = cv2.split(depth1)
print(imageb.mean())
print(imageg.mean())
print(imager.mean())
print(predb.mean())
print(predg.mean())
print(predr.mean())
m=1.2
Ab=np.mean(predb)*m
Ag=np.mean(predg)*m
Ar=np.mean(predr)*m

print(Ab)
print(Ag)
print(Ar)
t_b = np.divide((np.subtract(predb, Ab)), (np.subtract(imageb, Ab))+EPS)*1.0
t_g = np.divide((np.subtract(predg, Ag)), (np.subtract(imageg, Ag))+EPS)*1.0
t_r = np.divide((np.subtract(predr, Ar)), (np.subtract(imager, Ar))+EPS)*1.0

t_b = np.maximum(t_b,0)
t_g = np.maximum(t_g,0)
t_r = np.maximum(t_r,0)

betab = -np.divide(np.log(t_b+EPS), depthb/255.0*10.0+EPS)
betag = -np.divide(np.log(t_g+EPS), depthg/255.0*10.0+EPS)
betar = -np.divide(np.log(t_r+EPS), depthr/255.0*10.0+EPS)


betab = np.mean((betab-betab.min())/(betab.max()-betab.min()))*3.0
betag = np.mean((betag-betag.min())/(betag.max()-betag.min()))*3.0
betar = np.mean((betar-betar.min())/(betar.max()-betar.min()))*3.0

print(betab)
print(betag)
print(betar)
betadepthb =- np.multiply(betab, depthb/255.0*10.0)
betadepthg =- np.multiply(betag, depthg/255.0*10.0)
betadepthr = -np.multiply(betar, depthr/255.0*10.0)

newb = np.add(np.multiply(imageb, np.exp(betadepthb)) ,np.multiply(Ab, (np.subtract(1, np.exp(betadepthb)))))
newg = np.add(np.multiply(imageg, np.exp(betadepthg)) ,np.multiply(Ag, (np.subtract(1, np.exp(betadepthg)))))
newr = np.add(np.multiply(imager, np.exp(betadepthr)) ,np.multiply(Ar, (np.subtract(1, np.exp(betadepthr)))))
merge = cv2.merge([newb, newg, newr])
cv2.imshow('newb',newb/255.0)
cv2.imshow('newg',newg/255.0)
cv2.imshow('newr',newr/255.0)
cv2.imshow('merge',merge/255.0)
cv2.imwrite('3.png',merge)
cv2.waitKey(0)
cv2.destroyAllWindows()
