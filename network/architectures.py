# -*- coding: utf-8 -*-
"""
Network architetures assembled from .modules

@author: H. Huang
"""
#%%
from .modules import Encoder,BottleNeck,Decoder
import torch
from torch import nn
#%%Aux functions
def formalizeNormMethodName(n):
    if 'batch' in n.lower():
        res = 'BatchNorm'
    elif 'instance' in n.lower():
        res = 'InstanceNorm'
    else:
        raise(Exception(f'{n} normalization is not implemented'))
    return res

def formalizeActivationName(n):
    if 'relu' in n.lower():
        res = 'LeakyReLU'
    elif 'sigmoid' in n.lower():
        res = 'Sigmoid'
    elif 'tanh' in n.lower():
        res = 'Tanh'
    else:
        raise(Exception(f'{n} normalization is not implemented'))
    return res

def initWeights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            print('.',end='')


#%% Image-to-image multi-level encoder-decoder
class Im2ImMultiLevel(torch.nn.Module):
    def __init__(self,dim, 
                     nLevel, dsFactor,
                     normMethod,activation,
                     inputChannel, outputChannel, 
                     bottleNeckChennel ,
                     channelNum, kernelSize):
        super().__init__()
        normMethod = formalizeNormMethodName(normMethod)
        activation = formalizeActivationName(activation)
        dsModuleList = []
        encoderList = []
        bottleNeckList = []
        decoderList =[]
        usModuleList = []
        for nL in range(nLevel):
            # downsampling module
            if dsFactor[nL] > 1:
                #dsModule = torch.nn.Upsample(scale_factor=1/dsFactor[nL],mode='bilinear' if dim==2 else 'trilinear', align_corners=True)
                dsModule = torch.nn.Upsample(scale_factor=1/dsFactor[nL],mode='nearest')
            else:
                dsModule = torch.nn.Identity()
            # encoder module
            encoderInputChannel = inputChannel + (nL>0)*outputChannel
            encoder = Encoder(dim, normMethod, activation, encoderInputChannel, channelNum[nL], kernelSize[nL])
            # bottle neck module 
            bottleNeckInputChannel = channelNum[nL][-1] if nL < 1 else channelNum[nL][-1] + channelNum[nL-1][-1]
            bottleNeckOutputChannel = channelNum[nL][-1]
            bottleNeck = BottleNeck(dim, normMethod, activation, 
                                    bottleNeckInputChannel, bottleNeckChennel[nL],bottleNeckOutputChannel)
            # decoder module
            decoder = Decoder(dim, normMethod, activation, outputChannel, channelNum[nL][::-1], kernelSize[nL][::-1])
            # upsampling module
            usModule = torch.nn.Upsample(scale_factor=dsFactor[nL],mode='bilinear' if dim==2 else 'trilinear', align_corners=True)
            # add module to list
            dsModuleList.append( dsModule )
            encoderList.append( encoder )
            bottleNeckList.append( bottleNeck )
            decoderList.append( decoder )
            usModuleList.append( usModule )
        self.dim = dim
        self.nLevel = nLevel
        self.dsFactor = dsFactor
        self.downsample = torch.nn.ModuleList(dsModuleList)
        self.encoder = torch.nn.ModuleList(encoderList)
        self.bottleNeck = torch.nn.ModuleList(bottleNeckList)
        self.decoder = torch.nn.ModuleList(decoderList)
        self.upsample = torch.nn.ModuleList(usModuleList)
        self.initialize()
    def initialize(self):
        print('initalizing network')
        self.apply(initWeights)
        print('\n')
    def forward(self,x):
        # creat list for storing intermediate outputs
        dsX = [None]*self.nLevel
        latent = [None]*self.nLevel
        out = [None]*self.nLevel
        # Step 1: downsample input as needed
        for nL in range(self.nLevel):
            dsX[nL] = self.downsample[nL](x)
        # step 2: pass through level 0 -- the coarsest level
        latent[0] = self.encoder[0](dsX[0])
        out[0] = self.decoder[0](
                        self.bottleNeck[0](latent[0])
                                )
        # step 3: pass through rest of levels -- higher res levels
        for nL in range(1,self.nLevel):
            tmpUpSampleFactor = self.dsFactor[nL-1]/self.dsFactor[nL]
            tmpUpsample = lambda x: torch.nn.functional.interpolate(x,scale_factor=tmpUpSampleFactor,
                                                                    mode='bilinear' if self.dim ==2 else 'trilinear',
                                                                    align_corners=True)
            prevOut = tmpUpsample(out[nL-1])
            currentInput = torch.cat( [dsX[nL],prevOut], dim=1)
            currentLatent = self.encoder[nL](currentInput)
            latentList = [tmpUpsample(latent[nL-1]), currentLatent]
            latent[nL] = torch.cat( latentList, dim=1)
            out[nL] = self.decoder[nL](
                            self.bottleNeck[nL](latent[nL])
                                      )
        # step 4: upsampel output from all levels
        for nL in range(self.nLevel):
            out[nL] = self.upsample[nL](out[nL])
        return sum(out)

