# -*- coding: utf-8 -*-
"""
Functions for data Augmentation

@author: H. Huang
"""
#%% import stuff
import torch
import torchvision.transforms.functional as tF
from torchvision.transforms import RandomCrop
from typing import List, Optional, Sequence, Tuple, Union

#%% Augmentation Object
class augmentator():
    def __init__(self,
                 doRigid=False,tRan=None,rRan=None,
                 doNoise=False,std=0.02,
                 doFlip=False,probH=0.5,probV=0.5,
                 doCrop=False,cropSize=[256,256]):
        if doRigid:
            self.rFunc = lambda x: randRigid(x,tRan,rRan)
        else:
            self.rFunc = lambda x: x
        if doNoise:
            self.nFunc = lambda x: addNoise(x, std)
        else:
            self.nFunc = lambda x: x
        if doFlip:
            self.fFunc = lambda x: randFlip(x, probH, probV)
        else:
            self.fFunc = lambda x: x
        if doCrop:
            self.cFunc = RandomCrop(cropSize)
        else:
            self.cFunc = lambda x: x
    def __call__(self,im,mask):
        numPos = mask.sum()
        stacked = torch.stack([torch.tensor(im),torch.tensor(mask)],dim=0)
        stacked = self.fFunc(stacked)
        stacked = self.rFunc(stacked)
        im,mask = self.cFunc(stacked)
        # while mask.sum() < numPos*0.8:
        #     im,mask = self.cFunc(stacked)
        im = self.nFunc(im).squeeze().numpy()
        mask = mask.squeeze().numpy()
        return im, mask
    
#%% random Rigid transformation
@torch.no_grad()
def randRigid(im:torch.tensor,tRan:List=None,rRan:List=None)->torch.tensor:
    doR = False if rRan is None else True
    doT = False if tRan is None else True
    vR = torch.rand(1).item()*(rRan[1]-rRan[0]) + rRan[0] if doR else 0.
    vT = torch.randint(tRan[0], tRan[1], (2,)).tolist() if doT else [0,0]
    im = tF.affine(im, vR, vT, 1, 0)
    # if doR:
    #     vR = torch.randint(rRan[0], rRan[1], (2,)).tolist()
    #     im = tF.affine(im, vR[0], (0,0), 1, 0)
    #     im = tF.affine(im.transpose(-1,-2), vR[1], (0,0), 1, 0).transpose(-1,-2)
    # if doT:
    #     vT = torch.randint(tRan[0], tRan[1], (2,)).tolist()
    #     im = tF.affine(im, 0, (vT[0],0),1, 0)
    #     im = tF.affine(im.transpose(-1,-2), 0, (vT[1],0), 1, 0).transpose(-1,-2)
    return im

#%% random noise augmentation
@torch.no_grad()
def addNoise(im:torch.tensor,std:float)->torch.tensor:
    noiseV = torch.randn(im.size())*std*im.float().mean()
    im += noiseV
    return im

#%% random horizontal and vertical flip
@torch.no_grad()
def randFlip(im:torch.tensor,probH:float,probV:float)->torch.tensor:
    hFlip = torch.rand(1)
    vFlip = torch.rand(1)
    if hFlip <= probH:
        im = tF.hflip(im)
    if vFlip <= probV:
        im = tF.vflip(im)
    return im
