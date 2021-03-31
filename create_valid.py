#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 08:25:52 2021

@author: amine
"""

import pandas as pd
train = pd.read_csv('data/SDP_train.csv')

test = pd.read_csv('data/SDP_test.csv')

train_0 = train[['citing_title', 'cited_title', 'citation_context', 'citation_class_label']]

test = test[['citing_title', 'cited_title', 'citation_context']]
test.to_csv('data/test.csv', index = False, header=True)


t_0 = train_0[train['citation_class_label']==0]
t_1 = train_0[train['citation_class_label']==1]
t_2 = train_0[train['citation_class_label']==2]
t_3 = train_0[train['citation_class_label']==3]
t_4 = train_0[train['citation_class_label']==4]
t_5 = train_0[train['citation_class_label']==5]




t_0_train = t_0.iloc[:int(len(t_0)*0.8)]
t_0_dev = t_0.iloc[int(len(t_0)*0.8):]
t_1_train = t_1.iloc[:int(len(t_1)*0.8)]
t_1_dev = t_1.iloc[int(len(t_1)*0.8):]
t_2_train = t_2.iloc[:int(len(t_2)*0.8)]
t_2_dev = t_2.iloc[int(len(t_2)*0.8):]
t_3_train = t_3.iloc[:int(len(t_3)*0.8)]
t_3_dev = t_3.iloc[int(len(t_3)*0.8):]
t_4_train = t_4.iloc[:int(len(t_4)*0.8)]
t_4_dev = t_4.iloc[int(len(t_4)*0.8):]
t_5_train = t_5.iloc[:int(len(t_5)*0.8)]
t_5_dev = t_5.iloc[int(len(t_5)*0.8):]


train = pd.concat([t_0_train, t_1_train, t_2_train, t_3_train, t_4_train, t_5_train])
dev = pd.concat([t_0_dev, t_1_dev, t_2_dev, t_3_dev, t_4_dev, t_5_dev])


train.to_csv('data/train.csv', index = False, header=True)
dev.to_csv('data/dev.csv', index = False, header=True)
    
        

