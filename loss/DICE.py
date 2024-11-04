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

class WeightedDiceLoss(torch.nn.Module):
    def __init__(self, empty_weight=0.3,
                 squared = True,
                 smooth = 1.):
        super().__init__()
        self.empty_weight = empty_weight
        self.squared = squared
        self.smooth = smooth  # Small constant for numerical stability

    def forward(self, logits, targets):
        batch_size = logits.size(0)
        dice_loss = 0.0
        reduce_axis = [-2,-1]
        for i in range(batch_size):
            pred = torch.sigmoid(logits[i])
            target = targets[i]
            
            intersection = (pred * target).sum(dim=reduce_axis)
            p = 2 if self.squared else 1
            denor = (pred**p).sum(dim=reduce_axis) + (target**p).sum(dim=reduce_axis)
            loss = 1. - (2 * intersection + self.smooth) / (denor + self.smooth)
            if target.sum() == 0:
                loss *= self.empty_weight
            # # Check if target is empty (all background)
            # if target.sum() == 0:
            #     # Assign lower weight for empty mask
            #     pred = 1-pred
            #     target = 1-target
            #     intersection = (pred * target).sum(dim=reduce_axis)
            #     p = 2 if self.squared else 1
            #     denor = (pred**p).sum(dim=reduce_axis) + (target**p).sum(dim=reduce_axis)
            #     loss = 1. - (2 * intersection + self.smooth) / (denor + self.smooth)
            #     loss *= self.empty_weight
            # else:
            #     # Standard Dice calculation for non-empty masks
            #     intersection = (pred * target).sum(dim=reduce_axis)
            #     p = 2 if self.squared else 1
            #     denor = (pred**p).sum(dim=reduce_axis) + (target**p).sum(dim=reduce_axis)
            #     loss = 1. - (2 * intersection + self.smooth) / (denor + self.smooth)
            
            # Accumulate weighted Dice loss
            dice_loss += loss
        
        return dice_loss / batch_size  # Average over batch