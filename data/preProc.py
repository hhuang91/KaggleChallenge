# -*- coding: utf-8 -*-
"""
Preprocessing of input data

@author: H. Huang
"""
#%%
import torch
#%%
mu = 0.3898
std = 0.2219

def reSize(x):
    x = torch.tensor(x).view(1,1,x.shape[-2:])
    x = torch.nn.functional.interpolate(x,[512,512],mode='nearest')
    return x.squeeze().numpy()

def preprocIm(im):
    im = im/255
    im = (im-mu)/std
    im = reSize(im)
    return im

def preprocMask(mask):
    mask = reSize(mask)
    pos = sum(mask>0.5)
    if pos < 1:
        norm_fact = 1
    else:
        neg = sum(mask<0.5)
        norm_fact = neg/pos
    return mask, norm_fact
