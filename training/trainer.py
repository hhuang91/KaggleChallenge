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
import contextlib

from ..loss.DICE import softBianaryDICE2d
from ..loss.BCE import weightedBCE2d
#%% class defination
class networkTrainer():
    def __init__(self,
                 device:str,
                 net:torch.nn.modules,
                 optimizer:torch.optim,
                 lossFunc,
                 continueTrain:bool ,xferLearn:bool,
                 EPOCHS:int,
                 dataSetLoader:dataLoader,
                 outDir:str,
                 rank:int = 0):
        self.rank = rank
        #self.lossFnc = ...
        self.net = net;
        self.optimizer = optimizer;
        self.lossFunc = lossFunc
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
                loss = self.forwardPass(data, train = True)
                datIter.set_description(f"Training, batch loss: {loss}")
                trainLoss += loss
            trainLoss /= len(self.trainLoader)
            valdnLoss = self.valdn()
            print(f"Epoch: {epoch}. Training Loss: {trainLoss}. Validation Loss: {valdnLoss}")
            self.trainLoss.append(trainLoss)
            self.valdnLoss.append(valdnLoss)
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
        datIter = self.valdnLoader#tqdm(self.valdnLoader,file=sys.stdout,desc="Validating")
        self.net.eval()
        for batch_idx, data in enumerate(datIter):
            loss = self.forwardPass(data, train = False)
            #datIter.set_description(f"Validation, batch loss: {loss}")
            valdnLoss += loss
        valdnLoss /= len(self.valdnLoader)
        return valdnLoss
    
    def forwardPass(self,data,train=True):
        image = data[0].to(self.device).view(-1,1,*data[0].shape[-2:])
        target = data[1].to(self.device).view(-1,1,*data[1].shape[-2:])
        bceWeight = data[2].to(self.device).view(-1)
        context = contextlib.nullcontext if train else torch.no_grad
        with context:
            pred = self.net(image)
            loss = self.loss(pred,target,bceWeight)
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.detach().cpu().numpy().squeeze().item()
        