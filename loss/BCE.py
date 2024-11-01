# -*- coding: utf-8 -*-
"""
Wrapper function for BCE losses

@author: H. Huang
"""
import torch
#%%

class weightedBCE2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,logit,target,weight):
        logit = logit.view(-1,1,*logit.shape[-2:])
        target = target.view(-1,1,*target.shape[-2:])
        weight = weight.view(-1)
        res = torch.nn.functional.binary_cross_entropy_with_logits(logit, target, weight=weight)
        return res