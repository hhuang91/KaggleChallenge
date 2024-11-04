# -*- coding: utf-8 -*-
"""
Warppers for easy use of some of the packages
especially for training the network

@author: H. Huang
"""
# external modules
import torch
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.io as sio
from glob import glob
import numpy as np
import pandas as pd
# custom modules
from data.dataHandler import dataLoader
from network.architectures import Im2ImMultiLevel, Im2ClassMultiLevel
from loss.DICE import softBianaryDICE2d, WeightedDiceLoss
from loss.BCE import weightedBCE2d
from training.trainer import networkTrainer
from data.preProc import preprocIm
from utils.RLE import encode_mask_for_submission
from data.postProc import postprocMask
#%% Training Segmentation
def TrainSegmentation(config_file,
                      device = 'cpu',
                      continueTrain = False,
                      xferLearn = False,
                      save_frequency = 1,
                      dice_bg_weight = None,
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
    if dice_bg_weight is None:
        dice_loss = softBianaryDICE2d()
    else:
        dice_loss = WeightedDiceLoss(empty_weight=dice_bg_weight)
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

#%% Test Segmemntation
@torch.no_grad()
def TestSegmentation(config_file,
                     net_state_dir,
                     output_dir,
                     save_name = 'test_result',
                     device = 'cpu',
                     data_type = 2,
                     empty_thresh = 2000):
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
        if pred.sum() < empty_thresh:
            pred.zero_().bool()
        target = target.bool().squeeze()
        FP[i] = (pred & ~target).sum().item()
        FN[i] = (~pred & target).sum().item()
        TP[i] = (pred & target).sum().item()
        TN[i] = (~pred & ~target).sum().item()
        if 2*TP[i] + FP[i] + FN[i] == 0:
            DICE[i] = 1
        else:
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
    save_file_name = os.path.join(output_dir,save_name+'.mat')
    sio.savemat(save_file_name, save_dict)
    
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
    train_empty_num = 'all'
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
    
#%% Test Classification
@torch.no_grad()
def TestClassification(config_file,
                     net_state_dir,
                     output_dir,
                     save_name = 'test_result',
                     device = 'cpu',
                     data_type = 2,
                     cutoff = 0):
    # check if output folder exists, if not create one
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # load json config file
    with open(config_file,'r') as f:
        config = json.load(f)
    # setup data loader
    data_dir = config['data_dir']
    csv_dir = config['csv_dir']
    batch_size = 1
    train_empty_num = 'all'
    aug_kwarg = config['augmentation']
    device = torch.device(device)
    data_loader = dataLoader(data_dir, csv_dir, batch_size, 
                             train_empty_num, device, aug_kwarg=aug_kwarg)
    # setup network
    network_Config = config['networkConfig']
    net = Im2ClassMultiLevel(**network_Config).to(device)
    state_dict = torch.load(net_state_dir)
    net.load_state_dict(state_dict['netState'])
    net.eval()
    loaders = data_loader.getLoader()
    test_dataloader = loaders[data_type]
    FP = torch.zeros(len(test_dataloader))
    FN = torch.zeros(len(test_dataloader))
    TP = torch.zeros(len(test_dataloader))
    TN = torch.zeros(len(test_dataloader))
    for i, data in enumerate(tqdm(test_dataloader)):
        image = data[0].float().to(device).view(-1,1,*data[0].shape[-2:])
        target = (data[1].sum()>0).to(device).view(-1,1)
        pred = net(image)
        pred = (pred.sum()>cutoff).squeeze()
        target = target.bool().squeeze()
        FP[i] = (pred & ~target).sum().item()
        FN[i] = (~pred & target).sum().item()
        TP[i] = (pred & target).sum().item()
        TN[i] = (~pred & ~target).sum().item()
    Accuracy = (TP.sum() + TN.sum())/(TP.sum() + TN.sum() + FP.sum() +FN.sum())
    Precision = TP.sum()/(TP.sum() + FP.sum())
    Recall = TP.sum()/(TP.sum() + FN.sum())
    Specificity = TN.sum()/(TN.sum() + FP.sum())
    save_dict = {
                 'TP':TP.numpy(), 'TN':TN.numpy(),
                 'FP':FP.numpy(), 'FN':FN.numpy(),
                 'Accuracy': Accuracy.numpy(),
                 'Precision': Precision.numpy(),
                 'Recall': Recall.numpy(),
                 'Specificity': Specificity.numpy()
                }
    save_file_name = os.path.join(output_dir,save_name+'.mat')
    sio.savemat(save_file_name, save_dict)

#%% Create submission file
@torch.no_grad()
def CreateSubmission(input_dir,
                     output_dir,
                     submission_file_name,
                     seg_config_file,
                     seg_net_state_dir,
                     cls_config_file,
                     cls_net_state_dir,
                     seg_pixel_cutoff = 5000,
                     cls_cutoff = -2,
                     device = 'cpu',):
    # load json config file
    with open(seg_config_file,'r') as f:
        seg_config = json.load(f)
    with open(cls_config_file,'r') as f:
        cls_config = json.load(f)
        
    # setup segmentation network
    seg_network_Config = seg_config['networkConfig']
    seg_net = Im2ImMultiLevel(**seg_network_Config).to(device)
    state_dict = torch.load(seg_net_state_dir)
    seg_net.load_state_dict(state_dict['netState'])
    seg_net.eval()
    
    # setup classification network
    cls_network_Config = cls_config['networkConfig']
    cls_net = Im2ClassMultiLevel(**cls_network_Config).to(device)
    state_dict = torch.load(cls_net_state_dir)
    cls_net.load_state_dict(state_dict['netState'])
    cls_net.eval()
    
    # setup input data
    im_list = glob(input_dir + '/*.tif')
    
    # prepare output
    res = {'img':[],'pixels':[]}
    
    # loop through all images
    for i, im_file_name in enumerate(tqdm(im_list)):
        # load and process image
        res['img'].append(i+1)
        im = plt.imread(im_file_name)
        im = preprocIm(im)
        im = torch.tensor(im).view(1,1,*im.shape[-2:]).float().to(device)
        # classification
        im_cls = cls_net(im)
        bp_exists = (im_cls>cls_cutoff).squeeze().cpu().numpy().astype(bool).item()
        if not bp_exists:
            res['pixels'].append(np.nan)
            continue
        # segmentation (only if bp_exists)
        pred = seg_net(im)
        mask = (pred>0).squeeze()
        if mask.sum() < seg_pixel_cutoff:
            res['pixels'].append(np.nan)
            continue
        # convert mask to rle (only if pass threshold test)
        mask = mask.cpu().float().numpy()
        mask = postprocMask(mask)>0.5
        mask_rle = encode_mask_for_submission(mask)
        res['pixels'].append(mask_rle)
    
    # save results
    res_df = pd.DataFrame(res)
    csv_file_name = os.path.join(output_dir,submission_file_name)
    res_df.to_csv(csv_file_name,index=False)