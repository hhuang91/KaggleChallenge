#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 00:47:51 2024

@author: huang
"""
import torch


class softBianaryDICE2d(torch.nn.Module):
    '''
    soft-dice loss for binaray segmentaiton in 2D
    '''
    def __init__(self,
                 squared = True,
                 smooth=1.):
        super().__init__()
        self.squared = squared
        self.smooth = smooth

    def forward(self, logit, target):
        '''
        inputs:
            logits: tensor of shape (N,1,H,W)
            label: tensor of shape(N,1,H,W)
        output:
            loss: tensor of shape(1)
        '''
        reduce_axis = [2,3]
        pred = torch.sigmoid(logit)
        intersection = (pred * target).sum(dim=reduce_axis)
        p = 2 if self.squared else 1
        denor = (pred**p).sum(dim=reduce_axis) + (target**p).sum(dim=reduce_axis)
        loss = 1. - (2 * intersection + self.smooth) / (denor + self.smooth)
        return loss.mean()
    