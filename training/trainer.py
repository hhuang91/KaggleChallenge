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
import random
import contextlib

# from ..loss.DICE import softBianaryDICE2d
# from ..loss.BCE import weightedBCE2d
#%% class defination
class networkTrainer():
    def __init__(self,
                 device:str,
                 net:torch.nn.modules,
                 optimizer:torch.optim,
                 lossFunc,
                 continueTrain:bool ,xferLearn:bool,
                 EPOCHS:int,
                 dataSetLoader,
                 outDir:str,
                 rank:int = 0):
        self.rank = rank
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
        returnIm = True
        for batch_idx, data in enumerate(datIter):
            if returnIm:
                loss, pred = self.forwardPass(data, train = False, returnIm=returnIm)
                pos = data[1].sum([-2,-1]).squeeze()
                if any(pos > 0):
                    index = pos.nonzero()[0]
                    self.plot(data[0][index], pred[index], data[1][index])
                    returnIm = False
            else:
                loss= self.forwardPass(data, train = False, returnIm=returnIm)
            #datIter.set_description(f"Validation, batch loss: {loss}")
            valdnLoss += loss
        valdnLoss /= len(self.valdnLoader)
        return valdnLoss
    
    def forwardPass(self,data,train=True,returnIm=False):
        image = data[0].float().to(self.device).view(-1,1,*data[0].shape[-2:])
        target = data[1].float().to(self.device).view(-1,1,*data[1].shape[-2:])
        bceWeight = data[2].float().to(self.device).view(-1)
        context = contextlib.nullcontext if train else torch.no_grad
        with context():
            pred = self.net(image)
            loss = self.lossFunc(pred,target,bceWeight)
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        loss = loss.detach().cpu().numpy().squeeze().item()
        if returnIm:
            return loss, (pred>0).detach().cpu().numpy()
        else:
            return loss
#%% Aux funcitons        
    def plot(self,im,pred,target):
        fig,axs = plt.subplots(1,3)
        axs[0].imshow(im.detach().cpu().squeeze().numpy(),cmap='gray')
        axs[0].set_title('image')
        axs[1].imshow(target.detach().cpu().squeeze().numpy(),cmap='gray')
        axs[1].set_title('target')
        axs[2].imshow(pred.squeeze(),cmap='gray')
        axs[2].set_title('prediction')
        plt.show()
        
    def saveState(self,state,stateFN,disp=True):
        torch.save(state,stateFN)
        if disp:
            print('-->current training state saved')
            
    def loadState(self,stateFN,ldOptm = True,partialLoad = False):
        state = torch.load(stateFN,self.device)
        cnnState = state['cnnState']
        try:
            self.cnn.load_state_dict(cnnState)
        except:
            cnnState = self.stateDictConvert(cnnState)
        finally:
            if partialLoad:
                cnnState = self.partialStateConvert(cnnState)
            self.cnn.load_state_dict(cnnState)
        self.cnn.load_state_dict(cnnState)
        print('loaded training state')
        if ldOptm:
            self.optimizer.load_state_dict(state['optimizerState'])
            print('loaded optimizer state')
            
    def stateDictConvert(self,DPstate):
        from collections import OrderedDict
        State = OrderedDict()
        for k, v in DPstate.items():
            name = k.replace("module.", "") # remove 'module.' of dataparallel
            State[name] = v
        return State
    
    def partialStateConvert(self,DPstate):
        cnnDict = self.cnn.state_dict()
        partialState = {k: v for k, v in DPstate.items() if k in cnnDict}
        cnnDict.update(partialState)
        return cnnDict
