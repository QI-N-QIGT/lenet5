#!/usr/bin/env python
#coding:utf-8
from convLayer import *
from poolingLayer import *
from fullConnectionLayer import *
inputImage = [[1,2]]*28
class leNet5():
    def __init__(self):
        self.conv1 = convLayer([[32,32]],[[5,5]]*6).convolution(inputImage)  #Nx28x28的数组
        self.pool2 = poolingLayer(self.conv1,[2,2]).poolledLayer()           #self.pool2返回的是一个[][][]三维矩阵
        self.kernel3Size = [[5,5]]*6*16
        self.pool4 = poolingLayer(self.conv3,[2,2]).poolledLayer()



    def conv3(self,kernel3Size):
        conv3edImage = zeros((16,10,10))
        for i in range(0,16):
            for j in range(0,6):
                featureImage3 = convLayer([[14,14]],[[5,5]]*6).convolution(self.pool2[j])
                for m in range(6):
                    conv3edImage[i] = conv3edImage[i]+featureImage3[m]








