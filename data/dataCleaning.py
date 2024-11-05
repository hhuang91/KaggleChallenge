# -*- coding: utf-8 -*-
"""
Data cleaning script for picking out contradictory labels

based on https://fhtagn.net/prog/2016/08/19/kaggle-uns.html

@author: H. Huang
"""
#%%
import numpy as np
import skimage
import scipy.spatial.distance as spdist
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
#%%
data_dir = './data/train'
def compute_img_hist(img):
    # Divide the image in blocks and compute per-block histogram
    blocks = skimage.util.view_as_blocks(img, block_shape=(20, 20))
    img_hists = [np.histogram(block, bins=np.linspace(0, 1, 10))[0] for block in blocks]
    return np.concatenate(img_hists)

def dice_coefficient(Y_pred, Y):
    """
    This works for one image
    http://stackoverflow.com/a/31275008/116067
    """
    denom = (np.sum(Y_pred == 1) + np.sum(Y == 1))
    if denom == 0:
        # By definition, see https://www.kaggle.com/c/ultrasound-nerve-segmentation/details/evaluation
        return 1
    else:
        return 2 * np.sum(Y[Y_pred == 1]) / float(denom)
#%%
def load_patient(pid,df):
    p_df = df.loc[df['subject'] == pid].copy().reset_index(drop=True)
    imgs = []
    masks = []
    for idx in range(len(p_df)):
        subject = p_df.loc[idx,'subject']
        img = p_df.loc[idx,'img']
        if img == 5:
            1==1
        # assemble image directories
        im_dir = f'{data_dir}/{subject}_{img}.tif'
        mask_dir = f'{data_dir}/{subject}_{img}_mask.tif'
        # load images
        im = plt.imread(im_dir).astype(np.float32)
        mask = plt.imread(mask_dir)
        mask = (mask>0).astype(np.float32)
        imgs.append(im)
        masks.append(mask)
    return imgs, masks, p_df
#%%
def filter_images_for_patient(pid,df):
    imgs, masks, p_df = load_patient(pid,df)
    hists = np.array(list(map(compute_img_hist, imgs)))
    D = spdist.squareform(spdist.pdist(hists, metric='cosine'))
    
    # Used 0.005 to train at 0.67
    close_pairs = D + np.eye(D.shape[0]) < 0.008
    
    close_ij = np.transpose(np.nonzero(close_pairs))
    
    incoherent_ij = [(i, j) for i, j in close_ij if dice_coefficient(masks[i], masks[j]) < 0.2]
    incoherent_ij = np.array(incoherent_ij)
    
    valids = np.ones(len(imgs), dtype=bool)
    for i, j in incoherent_ij:
        if np.sum(masks[i]) == 0:
            valids[i] = False
        if np.sum(masks[j]) == 0:
            valids[i] = False
    p_df = p_df.loc[valids].copy().reset_index(drop=True)
    num_removed = (~valids).sum()
    return p_df,num_removed

#%%
train_df_nE = pd.read_csv('./csv/train_nonEmpty.csv')
train_df_e = pd.read_csv('./csv/train_empty.csv')
train_df = pd.concat([train_df_nE,train_df_e],ignore_index=True)
#%%
iterObj = tqdm(range(1,42),desc='removing invalid data: ')
for n, pid in enumerate(iterObj):
    p_df, num_removed = filter_images_for_patient(pid, train_df)
    if n == 0:
        new_train_df = p_df
    else:
        new_train_df = pd.concat([new_train_df,p_df],ignore_index=True)
    iterObj.set_description(f"data removed -- {num_removed}")
#%%
def nonEmpty(row):
    arg1 = train_df_nE['subject'] == row['subject']
    arg2 = train_df_nE['img'] == row['img']
    return any(arg1 & arg2)
def empty(row):
    arg1 = train_df_e['subject'] == row['subject']
    arg2 = train_df_e['img'] == row['img']
    return any(arg1 & arg2)
new_train_df_nE = new_train_df.loc[new_train_df.apply(nonEmpty,axis=1)].copy().reset_index(drop=True)
new_train_df_e = new_train_df.loc[new_train_df.apply(empty,axis=1)].copy().reset_index(drop=True)
#%%
valdn_df_nE = pd.read_csv('./csv/valdn_nonEmpty.csv')
valdn_df_e = pd.read_csv('./csv/valdn_empty.csv')
valdn_df = pd.concat([valdn_df_nE,valdn_df_e],ignore_index=True)
iterObj = tqdm(range(42,47),desc='removing invalid data: ')
for n, pid in enumerate(iterObj):
    p_df, num_removed = filter_images_for_patient(pid, valdn_df)
    if n == 0:
        new_valdn_df = p_df
    else:
        new_valdn_df = pd.concat([new_valdn_df,p_df],ignore_index=True)
    iterObj.set_description(f"data removed -- {num_removed}")

def nonEmpty(row):
    arg1 = valdn_df_nE['subject'] == row['subject']
    arg2 = valdn_df_nE['img'] == row['img']
    return any(arg1 & arg2)
def empty(row):
    arg1 = valdn_df_e['subject'] == row['subject']
    arg2 = valdn_df_e['img'] == row['img']
    return any(arg1 & arg2)
new_valdn_df_nE = new_valdn_df.loc[new_valdn_df.apply(nonEmpty,axis=1)].copy().reset_index(drop=True)
new_valdn_df_e = new_valdn_df.loc[new_valdn_df.apply(empty,axis=1)].copy().reset_index(drop=True)
new_valdn_df_e.to_csv('./csv/valdn_empty.csv',index=False)
new_valdn_df_nE.to_csv('./csv/valdn_nonEmpty.csv',index=False)
#%%
test_df_nE = pd.read_csv('./csv/test_nonEmpty.csv')
test_df_e = pd.read_csv('./csv/test_empty.csv')
test_df = pd.concat([test_df_nE,test_df_e],ignore_index=True)
iterObj = tqdm(range(47,48),desc='removing invalid data: ')
for n, pid in enumerate(iterObj):
    p_df, num_removed = filter_images_for_patient(pid, test_df)
    if n == 0:
        new_test_df = p_df
    else:
        new_test_df = pd.concat([new_test_df,p_df],ignore_index=True)
    iterObj.set_description(f"data removed -- {num_removed}")

def nonEmpty(row):
    arg1 = test_df_nE['subject'] == row['subject']
    arg2 = test_df_nE['img'] == row['img']
    return any(arg1 & arg2)
def empty(row):
    arg1 = test_df_e['subject'] == row['subject']
    arg2 = test_df_e['img'] == row['img']
    return any(arg1 & arg2)
new_test_df_nE = new_test_df.loc[new_test_df.apply(nonEmpty,axis=1)].copy().reset_index(drop=True)
new_test_df_e = new_test_df.loc[new_test_df.apply(empty,axis=1)].copy().reset_index(drop=True)
new_test_df_e.to_csv('./csv/test_empty.csv',index=False)
new_test_df_nE.to_csv('./csv/test_nonEmpty.csv',index=False)