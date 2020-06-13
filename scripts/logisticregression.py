from numpy import *
from os import listdir


def loadData(direction):
    trainfileList=listdir(direction)
    m=len(trainfileList)
    dataArray= zeros((m,1024))
    labelArray= zeros((m,1))
    for i in range(m):
        returnArray=zeros((1,1024))
        filename=trainfileList[i]
        fr=open('%s/%s' %(direction,filename))
        for j in range(32):
            lineStr=fr.readline()
            for k in range(32):
                returnArray[0,32*j+k]=int(lineStr[k])
        dataArray[i,:]=returnArray 
    
        filename0=filename.split('.')[0]
        label=filename0.split('_')[0]
        labelArray[i]=int(label) 
    return dataArray,labelArray
    
def sigmoid(inX):
    return 1.0/(1+exp(-inX))


def gradAscent(dataArray,labelArray,alpha,maxCycles):
    dataMat=mat(dataArray)    #size:m*n
    labelMat=mat(labelArray)      #size:m*1
    m,n=shape(dataMat)
    weigh=ones((n,1)) 
    for i in range(maxCycles):
        h=sigmoid(dataMat*weigh)
        error=labelMat-h    #size:m*1
        weigh=weigh+alpha*dataMat.transpose()*error
    return weigh

def classfy(testdir,weigh):
    dataArray,labelArray=loadData(testdir)
    dataMat=mat(dataArray)
    labelMat=mat(labelArray)
    h=sigmoid(dataMat*weigh)  #size:m*1
    m=len(h)
    error=0.0
    for i in range(m):
        if int(h[i])>0.5:
            print int(labelMat[i]),'is classfied as: 1'
            if int(labelMat[i])!=1:
                error+=1
                print 'error'
        else:
            print int(labelMat[i]),'is classfied as: 0'
            if int(labelMat[i])!=0:
                error+=1
                print 'error'
    print 'error rate is:','%.4f' %(error/m)
                
def digitRecognition(trainDir,testDir,alpha=0.07,maxCycles=10):
    data,label=loadData(trainDir)
    weigh=gradAscent(data,label,alpha,maxCycles)
    classfy(testDir,weigh)
