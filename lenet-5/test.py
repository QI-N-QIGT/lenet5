from lenet5 import *


print(fullConnect6())
def testPoolingOutput():
    '''
    测试pooling层and s2层，successful
    :return:
    '''
    outPutc1 = convLayer1()
    print(outPutc1[1])
    print("********************************************************")
    # input = [[1,2,3,4],[1,2,3,4],[6,7,8,9],[3,4,5,3]]
    # print(poolingFunction(input,1,4,2))
    print(poolingFunction(outPutc1,1,28,2))
    print("********************************************************")
    print(sLayer2())

def testConvLayer1():
    '''
    #测试卷积层1
    :return:
    '''
    print(convLayer1())

def testConvFunction():
    '''
    #测试卷积函数
    :return:
    '''
    test_img = imgRead()
    test_kernel = generateKernel()
    test_out = convFunction(test_img,test_kernel,1,32,5,1)
    print(test_img)
    print(test_out)
    print(np.array(test_out).shape)

def testConvFunction1():
    '''
    #用简单地输入，已知结果的输入来测试卷积函数，successful
    :return:
    '''
    test_img = [[1,1,1],[1,1,1],[2,1,1]]
    test_kernel = [[1,1],[1,0]]
    test_out = np.array(convFunction(test_img,test_kernel,1,3,2,1))
    print(test_out)
    print(np.array(test_out).shape)
