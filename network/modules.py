# -*- coding: utf-8 -*-
"""
Customized modules for building network

@author: huang
"""
#%%
import torch
#%% Aux functions
def getPadSize(kSize,stride=1,inputSize=None):
    if stride != 1:
        assert inputSize is not None
    if inputSize is None:
        inputSize = 0
    p = ( kSize - stride + (stride-1)*inputSize )/2
    return int(p)

#%% Building blocks
class Encoder(torch.nn.Module):
    def __init__(self,dim,normMethod,activation,inputChannel,channelNum,kernelSize,dropout = None):
        super().__init__()
        conv = eval(f'torch.nn.Conv{dim}d')
        pool = eval(f'torch.nn.MaxPool{dim}d')
        norm = eval(f'torch.nn.{normMethod}{dim}d')
        actv = eval(f'torch.nn.{activation}')
        if dropout is not None:
            drp = eval(f'torch.nn.Dropout{dim}d')
        moduleList = []
        for n,(cNum,kSize) in enumerate(zip(channelNum,kernelSize)):
            if n<1:
                inputC = inputChannel
            else:
                inputC = channelNum[n-1]
            pad = getPadSize(kSize)
            convLayer = conv(inputC, cNum, kSize, stride = 1, padding = pad , padding_mode='replicate')
            actvLayer = actv()
            poolLayer = pool(2)
            normLayer = norm(cNum)
            moduleList += [convLayer,actvLayer,poolLayer,normLayer]
            if dropout is not None:
                moduleList += [drp(dropout[n])]
        self.net = torch.nn.Sequential(*moduleList)
    def forward(self,x):
        return self.net(x)


class BottleNeck(torch.nn.Module):
    def __init__(self,dim,normMethod,activation,inputChannel,channelNum,outputChannel,dropout=None):
        super().__init__()
        conv = eval(f'torch.nn.Conv{dim}d')
        norm = eval(f'torch.nn.{normMethod}{dim}d')
        actv = eval(f'torch.nn.{activation}')
        moduleList=[]
        for n,cNum in enumerate(channelNum):
            if n<1:
                inputC = inputChannel
            else:
                inputC = channelNum[n-1]
            convLayer = conv(inputC, cNum, 1, stride = 1)
            actvLayer = actv()
            normLayer = norm(cNum)
            moduleList += [convLayer,actvLayer,normLayer]
            if dropout is not None:
                moduleList += [torch.nn.Dropout(dropout[n])]
        # extra layer to convert channel size to specification
        convLayer = conv(cNum, outputChannel, 1, stride = 1)
        actvLayer = actv()
        normLayer = norm(outputChannel)
        moduleList += [convLayer,actvLayer,normLayer]
        self.net = torch.nn.Sequential(*moduleList)
    def forward(self,x):
        return self.net(x)

class Decoder(torch.nn.Module):
    def __init__(self,dim,normMethod,activation,outputChannel,channelNum,kernelSize,dropout=None):
        super().__init__()
        conv = eval(f'torch.nn.Conv{dim}d')
        norm = eval(f'torch.nn.{normMethod}{dim}d')
        actv = eval(f'torch.nn.{activation}')
        if dropout is not None:
            drp = eval(f'torch.nn.Dropout{dim}d')
        moduleList = []
        for n,(cNum,kSize) in enumerate(zip(channelNum,kernelSize)):
            if n >= len(channelNum) - 1:
                outputC = outputChannel
            else:
                outputC = channelNum[n+1]
            pad = getPadSize(kSize)
            convLayer = conv(cNum, outputC, kSize, stride = 1, padding = pad , padding_mode='replicate')
            actvLayer = actv()
            normLayer = norm(outputC)
            upLayer   = torch.nn.Upsample(scale_factor=2,mode='bilinear' if dim==2 else 'trilinear', align_corners=True)
            moduleList += [convLayer,actvLayer,normLayer,upLayer]
            if dropout is not None:
                moduleList += [drp(dropout[n])]
        self.net = torch.nn.Sequential(*moduleList)
    def forward(self,x):
        return self.net(x)

class MLP(torch.nn.Module):
    def __init__(self,normMethod,activation,inputFeature,featureNum,dropout,outputFeature):
        super().__init__()
        linear = torch.nn.Linear
        norm = eval(f'torch.nn.{normMethod}1d')#torch.nn.Identity#torch.nn.LayerNorm#
        actv = eval(f'torch.nn.{activation}')
        moduleList = []
        for n,fNum in enumerate(featureNum):
            if n<1:
                inputF = inputFeature
            else:
                inputF = featureNum[n-1]
            linLayer = linear(inputF, fNum)
            actvLayer = actv()
            normLayer = norm(fNum)
            moduleList += [linLayer,actvLayer,normLayer,torch.nn.Dropout(dropout[n])]
        # final layer to convert size to 1
        linLayer = linear(fNum, outputFeature)
        actvLayer = actv()
        normLayer = norm(outputFeature)
        moduleList += [linLayer,actvLayer,normLayer]
        self.net = torch.nn.Sequential(*moduleList)
    def forward(self,x):
        return self.net(x)