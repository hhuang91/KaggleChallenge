# -*- coding: utf-8 -*-
"""
Object to handle object training

@author: H. Huang
"""
#%% Import
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import scipy.io
import os
import sys
import matplotlib.pyplot as plt
from ..data.dataHandler import dataLoader
import random

#%% class defination
class trainNTest():
    def __init__(self,
                 device:str,
                 net:torch.nn.modules,
                 optimizer:torch.optim,
                 continueTrain:bool ,xferLearn:bool,
                 EPOCHS:int,
                 dataSetLoader:dataLoader,
                 outDir:str,
                 rank:int = 0):
        self.rank = rank
        #self.lossFnc = ...
        self.net = net;
        self.optimizer = optimizer;
        self.outDir = outDir;
        for param_group in optimizer.param_groups:
            self.lr = param_group['lr']
        self.device = device;
        self.EPOCHS = EPOCHS
        self.trainLoader,self.valdnLoader,self.testLoader = dataSetLoader.getLoader()
        if not os.path.exists(self.outDir):
            os.makedirs(self.outDir)
        self.trainLoss = []
        self.valdnLoss = []
        self.lossDataFN = self.outDir+'/'+"lossData_LR_"+str(self.lr)+".mat"
        self.stateN = self.outDir+'/'+"state_lr_"+str(self.lr)+"_Epoch_"
        self.startEpoch = 0;
        if xferLearn:
            print("Transfer Learning")
            self.lossDataFN = self.outDir+'/'+"lossDataXfer_LR_"+str(self.lr)+".mat"
            self.stateN = self.outDir+'/'+"stateXfer_lr_"+str(self.lr)+"_Epoch_"
            if not continueTrain:
                stateFN = self.outDir+'/' + "stateXfer"+".pth.tar"
                if os.path.exists(stateFN):
                    self.loadState(stateFN,ldOptm=False,partialLoad=True)
                    print(f"State loaded:{stateFN}")
                else:
                    raise Exception(f"Need state file named {stateFN}")
        if continueTrain:
            print("Continue Training")
            if os.path.exists(self.lossDataFN):
                self.trainLoss = scipy.io.loadmat(self.lossDataFN)["TrainLoss"]
                self.valdnLoss = scipy.io.loadmat(self.lossDataFN)["ValdnLoss"]
                print(f"Loss data loaded:{self.lossDataFN}")
            else:
                raise Exception("Loss Data file missing!")
            self.startEpoch = max(len(self.trainLoss),len(self.valdnLoss));
            stateFN = self.stateN + str(self.startEpoch-1)+".pth.tar";
            if os.path.exists(stateFN):
                self.loadState(stateFN)
                print(f"State loaded:{stateFN}")
            else:
                raise Exception("State file missing!")
    
    def train(self):
        print("Begin Training")
        print(f"Start from {self.startEpoch} Epoch")
        for epoch in range(self.startEpoch,self.EPOCHS):
            trainLoss = 0
            self.net.train()
            datIter = tqdm(self.trainLoader,file=sys.stdout,desc="Training")
            for batch_idx, data in enumerate(datIter):
                loss = self.computeLoss(data, train = True)
                datIter.set_description(f"Training, batch loss: {loss}")
                trainLoss += loss
            trainLoss /= len(self.trainLoader)
            valLoss = self.valdn()
            print(f"Epoch: {epoch}. Training Loss: {trainLoss}. Validation Loss: {valLoss}")
            self.trainLoss += trainLoss
            self.valdnLoss += valLoss
            if self.rank == 0:
                scipy.io.savemat(self.lossDataFN, {'TrainLoss':self.trainLoss,
                                                      'ValdnLoss':self.valdnLoss})
                state = {'netState' : self.net.state_dict(),
                          'optimizerState' : self.optimizer.state_dict()}
                stateFN = self.stateN+str(epoch)+".pth.tar";
                self.saveState(state,stateFN)
                
    @torch.no_grad()
    def valdn(self,valdnAtEpoch = -1):
        if valdnAtEpoch >= 0:
            stateFN = self.stateN+str(valdnAtEpoch)+".pth.tar"
            if os.path.exists(stateFN):
                self.loadState(stateFN,ldOptm = False,partialLoad=True)
                print(f'Previous state loaded @{valdnAtEpoch} Epoch for validation')
            else:
                raise Exception(f'State @{valdnAtEpoch} Epoch is not found.')
        print("Begin Validation")
        valdnLoss = 0
        datIter = tqdm(self.valdnLoader,file=sys.stdout,desc="Validating")
        self.net.eval()
        for batch_idx, data in enumerate(datIter):
            loss = self.computeLoss(data, train = False)
            datIter.set_description(f"Validation, batch loss: {loss}")
            valdnLoss += loss
        valdnLoss /= len(self.valdnLoader)
        return valdnLoss
    
    def computeLoss(self,data,train=True):
        return 
        