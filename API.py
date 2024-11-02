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
from network.architectures import Im2ImMultiLevel
from loss.DICE import softBianaryDICE2d
from loss.BCE import weightedBCE2d
from training.trainer import networkTrainer
#%% Training network using configuration json file
def Train(config_file,
          device = 'cpu',
          continueTrain = False,
          xferLearn = False,
          ):
    # load json config file
    with open(config_file,'r') as f:
        config = json.load(f)
    # setup data loader
    data_dir = config['data_dir']
    csv_dir = config['csv_dir']
    batch_size = config['batch_size']
    train_empty_num = config['empty_num']
    device = torch.device(device)
    data_loader = dataLoader(data_dir, csv_dir, batch_size, train_empty_num, device)
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
                                       out_dir)
    network_traininer.train()