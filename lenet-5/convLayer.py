#!/usr/bin/env python
#coding:utf-8
from numpy import *

randomConstantW = 1
randomConstantB = 1
class convLayer():
    def __init__(self,outputSize = [],kernelSize = []):     #这里的kernelsize传进来的[[]]*n,outsize[[]]*m的形式
        self.kernelW = []
        self.bias = []
        self.featureMap = []
        #卷积核参数的初始化
        for kernel in self.kernelSize:
            self.kernelW.append(random.uniform(0,randomConstantW,kernel))
            self.bias.append(random.uniform(0,randomConstantB))
        for map in self.outputSize:
            self.featureMap.append(zeros(map))
        self.kernelW = array(self.kernelW)
        self.featureMap = array(self.featureMap)

    def convolution(self,inputImage):
        inputImage=array(inputImage)
        Image_x,Image_y = inputImage.shape
        feature_num,kernel_x, kernel_y = self.kernelW.shape
        output_num,out_x,out_y = self.featureMap.shape
        sumValue = zeros((output_num))
        for m in range(output_num):
                for n in range(out_x):
                    for r in range(out_y):
                        for i in range(kernel_x):
                            for j in range(kernel_y):
                                sumValue[m]=sumValue[m] +self.kernelW[m][i][j]*self.inputImage[i+n][j+r]
                        self.featureMap[m][n][r]=sumValue/float(kernel_y*kernel_x)

        return self.featureMap
















