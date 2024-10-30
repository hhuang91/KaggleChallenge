# -*- coding: utf-8 -*-
"""
Warppers for easy use of some of the packages
especially for training the network

@author: H. Huang
"""
import torch
import torch.nn as nn

#%%
def Train(lr,deviceN,continueTrain=False,xferLearn=False,imSize=None,
          outSize=None,EPOCHS=None,batchSize=None,dataPath=None,
          outDir = None,numWorker=None,dispOn=False,loResLossWeight = 1.):
    
    device = torch.device(deviceN)
    if device.type =='cuda' and device.index == None:
        cnn = nn.DataParallel(DLVIFmap().to(device));
    else:
        cnn = DLVIFmap().to(device)
    optimizer = optim.Adam(cnn.parameters(), lr=lr)

    imSize=[64,64,64] if imSize is None else imSize
    outSize=[1,] if outSize is None else outSize
    EPOCHS=150 if EPOCHS is None else EPOCHS
    batchSize=4 if batchSize is None else batchSize
    datPath="./Data" if dataPath is None else dataPath
    outDir="./networkOutput/" if outDir is None else outDir
    numWorker = 0 if numWorker is None else numWorker
    dataLoader = hdf5MapLoader(datPath, batchSize, device.type =='cuda',numWorker)
    tNt = trainNTest(cnn,optimizer,device,
                 continueTrain,xferLearn,imSize,outSize,
                 EPOCHS,dataLoader,outDir,dispOn,loResLossWeight)
    tNt.train()


def Test(lr,deviceN,testAtEpoch,continueTrain=False,xferLearn=False,imSize=None,
          outSize=None,EPOCHS=None,batchSize=None,dataPath=None,
          outDir = None,numWorker=None,dispOn=False,rndSeed = 0,loResLossWeight = 1.):
    device = torch.device(deviceN)
    if device.type =='cuda' and device.index == None:
        cnn = nn.DataParallel(DLVIFmap().to(device));
    else:
        cnn = DLVIFmap().to(device)
    optimizer = optim.Adam(cnn.parameters(), lr=lr)

    imSize=[64,64,64,64,64,64] if imSize is None else imSize
    outSize=[1,] if outSize is None else outSize
    EPOCHS=150 if EPOCHS is None else EPOCHS
    batchSize=4 if batchSize is None else batchSize
    datPath="./Data" if dataPath is None else dataPath
    outDir="./networkOutput/" if outDir is None else outDir
    numWorker = 0 if numWorker is None else numWorker
    dataLoader = hdf5MapLoader(datPath, batchSize, device.type =='cuda',numWorker)
    tNt = trainNTest(cnn,optimizer,device,
                 continueTrain,xferLearn,imSize,outSize,
                 EPOCHS,dataLoader,outDir,dispOn,loResLossWeight)
    tNt.test(testAtEpoch,rndSeed)
