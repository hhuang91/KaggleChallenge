# -*- coding: utf-8 -*-
"""
Warppers for easy use of some of the packages
especially for training the network

@author: H. Huang
"""
import torch
import os
import json
from data.dataHandler import dataLoader
from network.architectures import Im2ImMultiLevel, Im2ClassMultiLevel, UNet
from loss.DICE import softBianaryDICE2d
from loss.BCE import weightedBCE2d
from training.trainer import networkTrainer
from tqdm import tqdm
import scipy.io as sio
#%% Training network using configuration json file
def TrainSegmentation(config_file,
                      device = 'cpu',
                      continueTrain = False,
                      xferLearn = False,
                      save_frequency = 1,
                      ):
    # load json config file
    with open(config_file,'r') as f:
        config = json.load(f)
    # setup data loader
    data_dir = config['data_dir']
    csv_dir = config['csv_dir']
    batch_size = config['batch_size']
    train_empty_num = config['empty_num']
    aug_kwarg = config['augmentation']
    device = torch.device(device)
    data_loader = dataLoader(data_dir, csv_dir, batch_size, 
                             train_empty_num, device, aug_kwarg=aug_kwarg)
    # setup network
    network_Config = config['networkConfig']
    net = Im2ImMultiLevel(**network_Config).to(device)
    # setup optimizer
    lr = config['lr']
    optimizer = torch.optim.AdamW(net.parameters(),lr = lr)
    # set up loss function
    dice_loss = softBianaryDICE2d()
    bce_loss = weightedBCE2d()
    def loss_function(logit, target, weight):
        d = dice_loss(logit,target) * config['lossConfig']['DICE']
        b = bce_loss(logit,target,weight) * config['lossConfig']['BCE']
        return d+b
    # set up trainer
    EPOCHS = config['EPOCHS']
    out_dir= config['out_dir']
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f)
    network_traininer = networkTrainer(device, 
                                       net, optimizer, loss_function, 
                                       continueTrain, xferLearn, 
                                       EPOCHS, 
                                       data_loader, 
                                       out_dir,
                                       save_frequency=save_frequency)
    network_traininer.train()
#%% Train Segmentation with standard U-Net
def TrainSegmentationUnet(config_file,
                      device = 'cpu',
                      continueTrain = False,
                      xferLearn = False,
                      save_frequency = 1,
                      ):
    # load json config file
    with open(config_file,'r') as f:
        config = json.load(f)
    # setup data loader
    data_dir = config['data_dir']
    csv_dir = config['csv_dir']
    batch_size = config['batch_size']
    train_empty_num = config['empty_num']
    aug_kwarg = config['augmentation']
    device = torch.device(device)
    data_loader = dataLoader(data_dir, csv_dir, batch_size, 
                             train_empty_num, device, aug_kwarg=aug_kwarg)
    # setup network
    #network_Config = config['networkConfig']
    net = UNet(in_channels=1,init_features=8).to(device)
    # setup optimizer
    lr = config['lr']
    optimizer = torch.optim.AdamW(net.parameters(),lr = lr)
    # set up loss function
    dice_loss = softBianaryDICE2d()
    bce_loss = weightedBCE2d()
    def loss_function(logit, target, weight):
        d = dice_loss(logit,target) * config['lossConfig']['DICE']
        b = bce_loss(logit,target,weight) * config['lossConfig']['BCE']
        return d+b
    # set up trainer
    EPOCHS = config['EPOCHS']
    out_dir= config['out_dir']
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f)
    network_traininer = networkTrainer(device, 
                                       net, optimizer, loss_function, 
                                       continueTrain, xferLearn, 
                                       EPOCHS, 
                                       data_loader, 
                                       out_dir,
                                       save_frequency=save_frequency)
    network_traininer.train()
#%% Train Clasification
def TrainClassification(config_file,
                      device = 'cpu',
                      continueTrain = False,
                      xferLearn = False,
                      save_frequency = 1,
                      ):
    # load json config file
    with open(config_file,'r') as f:
        config = json.load(f)
    # setup data loader
    data_dir = config['data_dir']
    csv_dir = config['csv_dir']
    batch_size = config['batch_size']
    train_empty_num = 'all'#config['empty_num']
    aug_kwarg = config['augmentation']
    device = torch.device(device)
    data_loader = dataLoader(data_dir, csv_dir, batch_size, 
                             train_empty_num, device,
                             aug_kwarg=aug_kwarg,
                             shuffle_valdn=True)
    # setup network
    network_Config = config['networkConfig']
    net = Im2ClassMultiLevel(**network_Config).to(device)
    # setup optimizer
    lr = config['lr']
    optimizer = torch.optim.AdamW(net.parameters(),lr = lr)
    # set up loss function
    weight = len(data_loader.trainLoader.sampler.indx2)/len(data_loader.trainLoader.sampler.indx1)
    bce = torch.nn.BCEWithLogitsLoss( torch.tensor(weight).to(device) )
    def loss_function(logit, target, w):
        target = target.sum([-3,-2,-1])>0
        loss = bce(logit.squeeze(), target.float())
        return loss
    # set up trainer
    EPOCHS = config['EPOCHS']
    out_dir= config['out_dir']
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f)
    network_traininer = networkTrainer(device, 
                                       net, optimizer, loss_function, 
                                       continueTrain, xferLearn, 
                                       EPOCHS, 
                                       data_loader, 
                                       out_dir,
                                       save_frequency=save_frequency,
                                       do_plot=False)
    network_traininer.train()
    
#%% Test Segmemntation
@torch.no_grad()
def TestSegmentation(config_file,
                     net_state_dir,
                     output_dir,
                     device = 'cpu'):
    # check if output folder exists, if not create one
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # load json config file
    with open(config_file,'r') as f:
        config = json.load(f)
    # setup data loader
    data_dir = config['data_dir']
    csv_dir = config['csv_dir']
    batch_size = 1#config['batch_size']
    train_empty_num = config['empty_num']
    aug_kwarg = config['augmentation']
    device = torch.device(device)
    data_loader = dataLoader(data_dir, csv_dir, batch_size, 
                             train_empty_num, device, aug_kwarg=aug_kwarg)
    # setup network
    network_Config = config['networkConfig']
    net = Im2ImMultiLevel(**network_Config).to(device)
    state_dict = torch.load(net_state_dir)
    net.load_state_dict(state_dict['netState'])
    net.eval()
    _,_,test_dataloader = data_loader.getLoader()
    FP = torch.zeros(len(test_dataloader))
    FN = torch.zeros(len(test_dataloader))
    TP = torch.zeros(len(test_dataloader))
    TN = torch.zeros(len(test_dataloader))
    DICE = torch.zeros(len(test_dataloader))
    Accuracy = torch.zeros(len(test_dataloader))
    Precision = torch.zeros(len(test_dataloader))
    Recall = torch.zeros(len(test_dataloader))
    for i, data in enumerate(tqdm(test_dataloader)):
        image = data[0].float().to(device).view(-1,1,*data[0].shape[-2:])
        target = data[1].float().to(device).view(-1,1,*data[1].shape[-2:])
        pred = net(image)
        pred = (pred>0).squeeze()
        target = target.bool().squeeze()
        FP[i] = (pred & ~target).sum().item()
        FN[i] = (~pred & target).sum().item()
        TP[i] = (pred & target).sum().item()
        TN[i] = (~pred & ~target).sum().item()
        DICE[i] = (2*TP[i])/(2*TP[i] + FP[i] + FN[i]) 
        Accuracy[i] = (TP[i] + TN[i])/(TP[i] + TN[i] + FP[i] +FN[i])
        Precision[i] = TP[i]/(TP[i] + FP[i])
        Recall[i] = TP[i]/(TP[i] + FN[i])
    save_dict = {
                 'TP':TP.numpy(), 'TN':TN.numpy(),
                 'FP':FP.numpy(), 'FN':FN.numpy(),
                 'DICE': DICE.numpy(),
                 'Accuracy': Accuracy.numpy(),
                 'Precision': Precision.numpy(),
                 'Recall': Recall.numpy()
                }
    save_file_name = os.path.join(output_dir,'test_result.mat')
    sio.savemat(save_file_name, save_dict)
#%%
@torch.no_grad()
def TestSegmentationUnet(config_file,
                     net_state_dir,
                     output_dir,
                     device = 'cpu',
                     data_type = 2):
    # check if output folder exists, if not create one
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # load json config file
    with open(config_file,'r') as f:
        config = json.load(f)
    # setup data loader
    data_dir = config['data_dir']
    csv_dir = config['csv_dir']
    batch_size = 1#config['batch_size']
    train_empty_num = config['empty_num']
    aug_kwarg = config['augmentation']
    device = torch.device(device)
    data_loader = dataLoader(data_dir, csv_dir, batch_size, 
                             train_empty_num, device, aug_kwarg=aug_kwarg)
    # setup network
    # network_Config = config['networkConfig']
    # net = Im2ImMultiLevel(**network_Config).to(device)
    net = UNet(in_channels=1,init_features=8).to(device)
    state_dict = torch.load(net_state_dir)
    net.load_state_dict(state_dict['netState'])
    net.eval()
    loaders = data_loader.getLoader()
    test_dataloader = loaders[data_type]
    FP = torch.zeros(len(test_dataloader))
    FN = torch.zeros(len(test_dataloader))
    TP = torch.zeros(len(test_dataloader))
    TN = torch.zeros(len(test_dataloader))
    DICE = torch.zeros(len(test_dataloader))
    Accuracy = torch.zeros(len(test_dataloader))
    Precision = torch.zeros(len(test_dataloader))
    Recall = torch.zeros(len(test_dataloader))
    for i, data in enumerate(tqdm(test_dataloader)):
        image = data[0].float().to(device).view(-1,1,*data[0].shape[-2:])
        target = data[1].float().to(device).view(-1,1,*data[1].shape[-2:])
        pred = net(image)
        pred = (pred>0).squeeze()
        target = target.bool().squeeze()
        FP[i] = (pred & ~target).sum().item()
        FN[i] = (~pred & target).sum().item()
        TP[i] = (pred & target).sum().item()
        TN[i] = (~pred & ~target).sum().item()
        DICE[i] = (2*TP[i])/(2*TP[i] + FP[i] + FN[i]) 
        Accuracy[i] = (TP[i] + TN[i])/(TP[i] + TN[i] + FP[i] +FN[i])
        Precision[i] = TP[i]/(TP[i] + FP[i])
        Recall[i] = TP[i]/(TP[i] + FN[i])
    save_dict = {
                 'TP':TP.numpy(), 'TN':TN.numpy(),
                 'FP':FP.numpy(), 'FN':FN.numpy(),
                 'DICE': DICE.numpy(),
                 'Accuracy': Accuracy.numpy(),
                 'Precision': Precision.numpy(),
                 'Recall': Recall.numpy()
                }
    save_file_name = os.path.join(output_dir,'test_result.mat')
    sio.savemat(save_file_name, save_dict)