# -*- coding: utf-8 -*-
"""
post-processing of network output
a.k.a inverse of preProc

@author: H. Huang
"""
#%%
import torch
import numpy as np
#%%
def resize(x):
    x = torch.tensor(x).view(1,1,*x.shape[-2:])
    x = torch.nn.functional.interpolate(x,[420,580],mode='nearest')
    return x.squeeze().numpy()

def postprocMask(mask):
    return resize(mask)
