#!/usr/bin/env python
#coding:utf-8
import cv2
import numpy as np
import math
def imgRead():
    '''
     #我现在 已经读取了一个image，并将其转化为灰度图
    :return: 32*32数组
    '''
    img = cv2.imread('D:/num8.jpg')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(32,32),interpolation=cv2.INTER_CUBIC)
#    cv2.imshow("img",img)
#    cv2.waitKey(0)
    if (img.all() == None):
        print("error")
    img = np.array(img)
#    print(img.shape)
    return img

def generateBias(biasSize):
    bias = np.random.random([biasSize,biasSize])
    return bias

def generateKernel(kernelNum):
    '''
    #产生卷积核，卷积核的大小5*5，共有kernelNum个
    :return: 返回值为(kernelNum,5,5)的tensor
    '''
    kernel = []
    for i in range(kernelNum):
        kernel.append(np.random.randint(0,1,(5,5)))
 #   print(kernel)
    return kernel



def convFunction(img,kernel,layerNum,imgSize,kernelSize,stride):
    '''
    #实现卷积功能
    :param img: 输入图像数组
    :param kernel: 卷积核张量
    :param layerNum: 第layerNum个卷积核，定位卷积核
    :param imgSize: 输入图像数组的大小
    :param kernelSize: 卷积核的大小
    :param stride: 步长
    :return: 卷积后的二维数组
    '''
    kernel = kernel[layerNum]
    img = img
    outSize = int((imgSize-kernelSize+1)/stride)                   #输出的大小
    bias = generateBias(outSize)
    convOutput = np.zeros([outSize,outSize])
    for i in range(outSize):
        for j in range(outSize):
            for m in range(kernelSize):
                for n in range(kernelSize):
                    convOutput[i][j]= convOutput[i][j] + (kernel[m][n]*img[i+m][j+n] + bias[i][j])/(kernelSize*kernelSize)
    return convOutput

def sigmoid(input,layerNum):
    '''
    sigmoid函数，现在仅限于求方阵,权重这里是随机的
    :param input: 输入三维张量[,,]
    :param layerNum:特征图第layerNum层数
    :return: 激活后的结果
    '''
    inputSize = np.array(input).shape[1]
    weights = np.random.random([inputSize,inputSize])
    output = np.zeros([inputSize,inputSize])
    for i in range(inputSize):
        for j in range(inputSize):
            strength = input[layerNum][i][j]*weights[i][j]
            output[i][j] = (1 / (1 + math.exp(-strength)))
    return output

def poolingFunction(inputArray,layerNum,inputSize,poolingSize):
    '''
    最大pooling函数
    :param inputArray: 输入的数组
    :param layerNum: 第layerNum层
    :param inputSize: 输入的大小
    :param poolingSize: pooling的大小
    :return: 经过pool之后的值
    '''
    imgC1 = inputArray[layerNum]
#    imgC1 = inputArray            #for easy test
    outputSize = int(inputSize/poolingSize)
    poolingOutput = np.zeros([outputSize,outputSize])
    maxTemp = 0
    for i in range(outputSize):
        for j in range(outputSize):
            for m in range(poolingSize):
                for n in range(poolingSize):
                    if (maxTemp < imgC1[2*i+m][2*j+n]):
                        maxTemp = imgC1[2*i+m][2*j+n]
                    poolingOutput[i][j] = maxTemp
    return poolingOutput

def convLayer1():
    '''
    调用卷积函数，实现卷积功能,并经过sigmoid激活函数
    :return: （6,28,28）tensor
    '''
    img1 = imgRead()
    kernel1 = generateKernel(6)
    imgSize1 = 32
    kernelSize1 = 5
    stride1 = 1
    convOutput1 = []
    convOutput = []
    for i in range(6):
        convOutput1.append(convFunction(img1,kernel1,i,imgSize1,kernelSize1,stride1))
    for i in range(6):
        convOutput.append(sigmoid(convOutput1,i))
    return convOutput

def sLayer2():
    '''
    输出6张特征图，大小为[14,14]
    :return: 三维的tensor[6,14,14]
    '''
    inputArray = convLayer1()
    outputS2 = []
    for i in range(6):
        outputS2.append(poolingFunction(inputArray,i,28,2))
    return outputS2

def convLayer3():
    '''
    C3层，测试 successful
    :return:产生[16,10,10]张量
    '''
    outputImg1 = np.zeros([16,10,10])
    outputImg = []
    temp = []
    img = sLayer2()
    kernel3 = generateKernel(96)
    for i in range(16):
        for j in range(6):
            temp.append(convFunction(img[j],kernel3,i*6+j,14,5,1))  #[16*6,10,10]
    for i in range(16):
        for j in range(6):
            for m in range(10):
                for n in range(10):
                    outputImg1[i][m][n] += temp[6*i+j][m][n]
    for i in range(16):
        outputImg.append(sigmoid(outputImg1, i))
    return outputImg

def sLayer4():
    '''
    输入是[16,10,10]张量      successful
    :return: [16,5,5]张量
    '''
    inputArray = convLayer3()
    outputS4 = []
    for i in range(16):
        outputS4.append(poolingFunction(inputArray,i,10,2))
    return outputS4

def convLayer5():
    '''
    C5层，输入的是[16,5,5]张量   测试successful
    :return:产生[120,1,1]张量
    '''
    outputImg1 = np.zeros([120,1,1])
    outputImg = []
    temp = []
    img = sLayer4()
    kernel3 = generateKernel(120)
    for i in range(120):
        for j in range(16):
            temp.append(convFunction(img[j],kernel3,i,5,5,1))  #[120*16,1,1]
    for i in range(120):
        for j in range(16):
                    outputImg1[i][0][0] += temp[j*16+i][0][0]
    for i in range(120):
        outputImg.append(sigmoid(outputImg1, i))
    return outputImg

def fullConnect6():

    weights = np.random.random([84*120,1,1])
    bias = np.random.random([84,1,1])
    img = convLayer5()
    result = []
    outputImg1 = np.zeros([84,1,1])
    outputImg = []
    for i in range(84):
        for j in range(120):
            result.append(convFunction(img[j],weights,i,1,1,1))
    for i in range(84):
        for j in range(120):
                    outputImg1[i][0][0] += result[i*120+j][0][0]
    outputImg1 = np.array(outputImg1) + bias
    for i in range(84):
        outputImg.append(sigmoid(outputImg1, i))
    print(np.array(outputImg).shape)
    return outputImg












