# -*- coding: utf-8 -*-
"""
Read CSV files and partition data by creating new CSV files

@author: H. Huang
"""
import pandas as pd


df_mask = pd.read_csv('./data/train_masks.csv')
empty = df_mask['pixels'].isna()
nonEmpty = ~df_mask['pixels'].isna()


df_empty = df_mask[empty].copy().reset_index(drop=True)
df_empty.to_csv('./csv/empty.csv',index=False)
df_empty = pd.read_csv('./csv/empty.csv')

df_nonEmpty = df_mask[nonEmpty].copy().reset_index(drop=True)
df_nonEmpty.to_csv('./csv/nonEmpty.csv',index=False)
df_nonEmpty = pd.read_csv('./csv/nonEmpty.csv')
#%%
def train_filter(row):
    return row not in [x for x in range(42,48)]
def valdn_filter(row):
    return row in [x for x in range(42,47)]
def test_filter(row):
    return row in [47]
#%%
df_train_nE = df_nonEmpty.loc[ df_nonEmpty['subject'].apply(train_filter) ].copy().reset_index(drop=True)
df_valdn_nE = df_nonEmpty.loc[ df_nonEmpty['subject'].apply(valdn_filter) ].copy().reset_index(drop=True)
df_test_nE = df_nonEmpty.loc[ df_nonEmpty['subject'].apply(test_filter) ].copy().reset_index(drop=True)
df_train_nE.to_csv('./csv/train_nonEmpty.csv',columns=['subject','img'],index=False)
df_valdn_nE.to_csv('./csv/valdn_nonEmpty.csv',columns=['subject','img'],index=False)
df_test_nE.to_csv('./csv/test_nonEmpty.csv',columns=['subject','img'],index=False)
#%%
df_train_e = df_empty.loc[ df_empty['subject'].apply(train_filter) ].copy().reset_index(drop=True)
df_valdn_e = df_empty.loc[ df_empty['subject'].apply(valdn_filter) ].copy().reset_index(drop=True)
df_test_e = df_empty.loc[ df_empty['subject'].apply(test_filter) ].copy().reset_index(drop=True)
df_train_e.to_csv('./csv/train_empty.csv',columns=['subject','img'],index=False)
df_valdn_e.to_csv('./csv/valdn_empty.csv',columns=['subject','img'],index=False)
df_test_e.to_csv('./csv/test_empty.csv',columns=['subject','img'],index=False)