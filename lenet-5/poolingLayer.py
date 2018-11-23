#!/usr/bin/env python
#coding:utf-8
from numpy import *

#self.featureMap[m][n][r]这是C1层输出的特征图
class poolingLayer():
    def __init__(self,inputFeature = [],poolSize = []):
        pass
    def poolledLayer(self):
        for i in range(6):
            poolingLayer.poolledXLayer(self,i)

    def poolledXLayer(self,layerNum):
        temp = 0
        featureNum,featureX,featureY = self.inputFeature.shape
        pooledFeature = zeros((featureNum,14,14))
        for x in range(0,featureX,2):
            for y in range(0,featureY,2):
                pooledFeature[layerNum][x/2][y/2] = self.toxtoMax([[self.inputFeature[layerNum][x][y],self.inputFeature[layerNum][x][y+1]],
                               [self.inputFeature[layerNum][x+1][y],self.inputFeature[layerNum][x+1][y+1]]])
        return pooledFeature

    def toxtoMax(self,minFeature=[]):
        temp = 0
        minFeature = array(minFeature)
        minFeature_x = minFeature.shape[0]
        minFeature_y = minFeature.shape[1]
        for i in minFeature_x:
            for j in minFeature_y:
                if temp < minFeature[i][j]:
                    temp = minFeature[i][j]
        return temp












