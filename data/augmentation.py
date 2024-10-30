# -*- coding: utf-8 -*-
"""
Functions for data Augmentation

@author: H. Huang
"""
#%% import stuff
import torch
import torchvision.transforms.functional as tF
from typing import List, Optional, Sequence, Tuple, Union

#%% Augmentation Object
class augmentator():
    def __init__(self,
                 doRigid=False,tRan=None,rRan=None,
                 doNoise=False,std=0.02,
                 doFlip=False,probH=0.5,probV=0.5):
        if doRigid:
            rFunc = lambda x: randRigid(x,tRan,rRan)
        else:
            rFunc = lambda x: x
        if doNoise:
            nFunc = lambda x: addNoise(x, std)
        else:
            nFunc = lambda x: x
        if doFlip:
            fFunc = lambda x: randFlip(x, probH, probV)
        else:
            fFunc = lambda x: x
        self.func = lambda x: fFunc( rFunc( nFunc(x) ) )
    def __call__(self,x):
        return self.func(x)
    
#%% random Rigid transformation
@torch.no_grad()
def randRigid(im:torch.tensor,tRan:List=None,rRan:List=None)->torch.tensor:
    doR = False if rRan is None else True
    doT = False if tRan is None else True
    if doR:
        vR = torch.rand(rRan[0], rRan[1], (3,1))
        im = tF.affine(im, vR[0], (0,0), 1, 0)
        im = tF.affine(im.transpose(-1,-2), vR[1], (0,0), 1, 0).transpose(-1,-2)
        im = tF.affine(im.transpose(-1,-3), vR[2], (0,0), 1, 0).transpose(-1,-3)
    if doT:
        vT = torch.rand(tRan[0], tRan[1], (3,1))
        im = tF.affine(im, 0, (vT[0],0),1, 0)
        im = tF.affine(im.transpose(-1,-2), 0, (vT[1],0), 1, 0).transpose(-1,-2)
        im = tF.affine(im.transpose(-1,-3), 0, (vT[2],0), 1, 0).transpose(-1,-3)
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
