# -*- coding: utf-8 -*-
"""
data loading objects for training and validating network

@author: H. Huang
"""
#%% import modules
import torch
from.augmentation import augmentator
from typing import List, Optional, Sequence, Tuple, Union

#%% dataset class, handling item (data) fetching
class dataSet(torch.utils.data.Dataset):
    def __init__(self,datPath:str, datKind:str, device:Union[torch.device,str],augKwarg:dict=None):
        # device to load data to
        self.device = device
        # path to data
        self.datPath = datPath
        # type: train/valdn/test
        self.datKind  = datKind
        if datKind == 'Train':
            self.augmentator = augmentator(**augKwarg)
    def __getitem__(self, idx:int) -> Tuple[torch.Tensor]:
        # get item function
        return
    def __len__(self) -> int:
        # get length of dataset
        return

#%% data loader class, handling setting up dataset objects
class dataLoader():
    def __init__(self, datPath:str, batchSize:int, device:Union[torch.device,str], numWorker:int = 0):
        self.device = torch.device(device)
        useCuda = device.type=='cuda'
        loaderKwagrs = {'batch_size':batchSize,'shuffle':True}
        if useCuda:
            loaderKwagrs.update({'num_workers': numWorker,'pin_memory': True})
        self.datPath = datPath
        self.trainDS = dataSet(datPath,'Train',device)
        self.valdnDS = dataSet(datPath,'Valdn',device)
        self.testDS = dataSet(datPath,'Test',device)
        self.trainLoader = torch.utils.data.DataLoader(self.trainDS,**loaderKwagrs)
        loaderKwagrs['shuffle'] = False
        self.valdnLoader = torch.utils.data.DataLoader(self.valdnDS,**loaderKwagrs)
        self.testLoader  = torch.utils.data.DataLoader(self.testDS, **loaderKwagrs)
    def getLoader(self) -> torch.utils.data.DataLoader:
        return self.trainLoader,self.valdnLoader,self.testLoader
