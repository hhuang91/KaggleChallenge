# -*- coding: utf-8 -*-
"""
Network architetures assembled from .modules

@author: H. Huang
"""
#%%
from .modules import Encoder,BottleNeck,Decoder,MLP
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
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        print('.',end='')
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)
        print('.',end='')
        
def getLatentImSize(imSize,ds,channelNum):
    nDS = len(channelNum)
    imSize = torch.tensor(imSize)
    dsSize = (imSize /ds / 2**nDS).tolist()
    return int(dsSize[0]*dsSize[1])

#%% Image-to-image multi-level encoder-decoder
# TODO: add skip connection
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
                        self.bottleNeck[0](latent[0]) + latent[0]
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
                            self.bottleNeck[nL](latent[nL]) + currentLatent
                                      )
        # step 4: upsampel output from all levels
        for nL in range(self.nLevel):
            out[nL] = self.upsample[nL](out[nL])
        return sum(out)
    
#%% Image-to-class multi-level encoder-decoder
class Im2ClassMultiLevel(torch.nn.Module):
    def __init__(self,dim,
                     imSize,
                     nLevel, dsFactor,
                     normMethod,activation,
                     inputChannel,
                     featureNum,
                     dropout,
                     bottleNeckChennel ,
                     channelNum, kernelSize,
                     bottleNeckDropout =None):
        super().__init__()
        normMethod = formalizeNormMethodName(normMethod)
        activation = formalizeActivationName(activation)
        dsModuleList = []
        encoderList = []
        bottleNeckList = []
        decoderList =[]
        for nL in range(nLevel):
            # downsampling module
            if dsFactor[nL] > 1:
                dsModule = torch.nn.Upsample(scale_factor=1/dsFactor[nL],mode='nearest')
            else:
                dsModule = torch.nn.Identity()
            # encoder module
            encoderInputChannel = inputChannel
            encoder = Encoder(dim, normMethod, activation, encoderInputChannel, channelNum[nL], kernelSize[nL])
            # bottle neck module 
            bottleNeckInputChannel = channelNum[nL][-1] if nL < 1 else channelNum[nL][-1] + channelNum[nL-1][-1]
            bottleNeckOutputChannel = channelNum[nL][-1]
            if bottleNeckDropout is not None:
                bottleNeckDropout_l = bottleNeckDropout[nL]
            else:
                bottleNeckDropout_l = None
            bottleNeck = BottleNeck(dim, normMethod, activation, 
                                    bottleNeckInputChannel, bottleNeckChennel[nL],bottleNeckOutputChannel,bottleNeckDropout_l)
            # MLP module
            latentImSize = getLatentImSize(imSize, dsFactor[nL] ,channelNum[nL])
            mlpInputFeature = latentImSize * bottleNeckOutputChannel
            mlp = MLP(normMethod,activation,mlpInputFeature,featureNum[nL],dropout[nL],1)
            # add module to list
            dsModuleList.append( dsModule )
            encoderList.append( encoder )
            bottleNeckList.append( bottleNeck )
            decoderList.append( mlp )
        self.dim = dim
        self.nLevel = nLevel
        self.dsFactor = dsFactor
        self.downsample = torch.nn.ModuleList(dsModuleList)
        self.encoder = torch.nn.ModuleList(encoderList)
        self.bottleNeck = torch.nn.ModuleList(bottleNeckList)
        self.decoder = torch.nn.ModuleList(decoderList)
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
        feature = self.bottleNeck[0](latent[0]) # + latent[0]
        feature = feature.view(feature.shape[0],-1)
        out[0] = self.decoder[0](feature)
        # step 3: pass through rest of levels -- higher res levels
        for nL in range(1,self.nLevel):
            tmpUpSampleFactor = self.dsFactor[nL-1]/self.dsFactor[nL]
            tmpUpsample = lambda x: torch.nn.functional.interpolate(x,scale_factor=tmpUpSampleFactor,
                                                                    mode='bilinear' if self.dim ==2 else 'trilinear',
                                                                    align_corners=True)
            currentInput = dsX[nL]
            currentLatent = self.encoder[nL](currentInput)
            latentList = [tmpUpsample(latent[nL-1]), currentLatent]
            latent[nL] = torch.cat( latentList, dim=1)
            
            feature = self.bottleNeck[nL](latent[nL]) # + currentLatent
            feature = feature.view(feature.shape[0],-1)
            out[nL] = self.decoder[nL](feature)
        return sum(out)

#%% U-NET
from collections import OrderedDict
class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
