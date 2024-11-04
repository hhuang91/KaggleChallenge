# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 02:29:55 2024

@author: hhuang91
"""

import numpy as np
#%%

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros((shape[0] * shape[1], shape[2]), dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = color
    return img.reshape(shape)

def encode_mask_for_submission(mask):
    '''
    Parameters
    ----------
    img : numpy.array
        must have dimension of [W,H,1].
        inital testing showed that masks read from tif must be transposed before encoding to match given rle
    Returns
    -------
    rle: str

    '''
    return rle_encode(mask.transpose(1,0)[...,None])