# -*- coding: utf-8 -*-
"""
Preprocessing of input data

@author: H. Huang
"""
#%%
import torch
import numpy as np
#%%
'''the values here is extracted purely from current training data'''
mu = 99.4071923455182#0.3898
std = 56.59492460345583#0.2219
#%%
def reSize(x):
    x = torch.tensor(x).view(1,1,*x.shape[-2:])
    x = torch.nn.functional.interpolate(x,[512,512],mode='nearest')
    return x.squeeze().numpy()

def preprocIm(im):
    #im = im/255 <-- not needed since we are performing Z-score normalization
    im = (im-mu)/std
    im = reSize(im)
    return im

def preprocMask(mask):
    mask = (reSize(mask)>0).astype(np.float32)
    pos = (mask>0.5).sum()
    if pos < 1:
        norm_fact = 1
    else:
        neg = (mask<0.5).sum()
        norm_fact = neg/pos
    return mask, norm_fact
