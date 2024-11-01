# -*- coding: utf-8 -*-
"""
data loading objects for training and validating network

@author: H. Huang
"""
#%% import modules
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from.augmentation import augmentator
from preProc import preprocIm, preprocMask
from typing import List, Optional, Sequence, Tuple, Union

#%%
class imbalanceSampler():
    def __init__(self,data1,data2,num_data2):
        self.indx1 = [x for x in range (len(data1))]
        self.indx2 = [x for x in range ( len(data1), len(data1)+len(data2) ) ]
        self.num2 = num_data2
    
    def __iter__(self):
        smp_indx_2 = np.random.choice(self.indx2, self.num2)
        total_indx = self.indx1 + smp_indx_2
        total_indx = np.random.choice(total_indx,len(total_indx))
        yield from total_indx
        
    def __len__(self):
        return len(self.indx1) + len(self.indx2)

#%% dataset class, handling item (data) fetching
class dataSet(torch.utils.data.Dataset):
    def __init__(self,data_dir: str,
                      df:pd.Dataframe,
                      data_kind:str,
                      aug_kwarg:dict = None,
                      set_len:int = None):
        # path to data
        self.data_dir = data_dir
        self.df = df
        # override length if specified
        self.set_len = set_len
        # set augmentator
        if data_kind.lower() == 'train':
            self.augmentator = augmentator(**aug_kwarg)
        else:
            self.augmentator = lambda x,y:(x,y)
            
    def __getitem__(self, idx:int) -> Tuple[torch.Tensor]:
        # read csv for info
        subject = self.df.loc[idx,'subject']
        img = self.df.loc[idx,'img']
        # assemble image directories
        im_dir = f'self.data_dir/{subject}_{img}.tif'
        mask_dir = f'self.data_dir/{subject}_{img}_mask.tif'
        # load images
        im = plt.imread(im_dir).astype(np.float32)
        mask = plt.imread(mask_dir).astype(np.float32)
        # preprocess images
        im = preprocIm(im)
        mask, norm_fact = preprocMask(mask)
        im,mask = self.augmentator(im,mask)
        return im, mask, norm_fact
    
    def __len__(self) -> int:
        if self.set_len is not None:
            return self.set_len
        return len(self.df)

#%% data loader class, handling setting up dataset objects
class dataLoader():
    def __init__(self, data_dir:str,
                 csv_dir:str,
                 batch_size:int, 
                 train_empty_num: int,
                 device:Union[torch.device,str], 
                 numWorker:int = 0,
                 aug_kwarg = None):
        # set cuda-related argument if necessary
        useCuda = device.type=='cuda'
        loaderKwagrs = {'batch_size':batch_size}
        if useCuda:
            loaderKwagrs.update({'num_workers': numWorker,'pin_memory': True})
            
        # load csv files
        train_df_nE = pd.read_csv(os.path.join(csv_dir,'train_nonEmpty.csv'))
        valdn_df_nE = pd.read_csv(os.path.join(csv_dir,'valdn_nonEmpty.csv'))
        test_df_nE = pd.read_csv(os.path.join(csv_dir,'test_nonEmpty.csv'))
        train_df_e = pd.read_csv(os.path.join(csv_dir,'train_empty.csv'))
        valdn_df_e = pd.read_csv(os.path.join(csv_dir,'valdn_empty.csv'))
        test_df_e = pd.read_csv(os.path.join(csv_dir,'test_empty.csv'))
        # combine csv files for easy data set management
        # selection of empty masks will ba handled by sampler
        train_df = pd.concat([train_df_nE,train_df_e],ignore_index=True)
        valdn_df = pd.concat([valdn_df_nE,valdn_df_e],ignore_index=True)
        test_df = pd.concat([test_df_nE,test_df_e],ignore_index=True)
        # get unique imbalanced sampler for training data
        ib_sampler = imbalanceSampler(train_df_nE, train_df_e, train_empty_num)
        # set data sets
        self.trainDS = dataSet(data_dir,train_df,'train',aug_kwarg, len(ib_sampler))
        self.valdnDS = dataSet(data_dir,valdn_df,'valdn')
        self.testDS = dataSet(data_dir,test_df,'test')
        self.trainLoader = torch.utils.data.DataLoader(self.trainDS,sampler=ib_sampler,**loaderKwagrs)
        loaderKwagrs['shuffle'] = False
        self.valdnLoader = torch.utils.data.DataLoader(self.valdnDS,**loaderKwagrs)
        self.testLoader  = torch.utils.data.DataLoader(self.testDS, **loaderKwagrs)
        
    def getLoader(self) -> torch.utils.data.DataLoader:
        return self.trainLoader,self.valdnLoader,self.testLoader
