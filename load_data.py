#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 14:03:38 2021

@author: amine
"""

import torch
from transformers import BertTokenizer
from torchtext import data

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length - 2]
    return tokens


init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)



CITING = data.Field(batch_first = True,
                  use_vocab = False,
                  tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_token_idx,
                  eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx)

CITED = data.Field(batch_first = True,
                  use_vocab = False,
                  tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_token_idx,
                  eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx)

CONTEXT = data.Field(batch_first = True,
                  use_vocab = False,
                  tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_token_idx,
                  eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx)


LABEL =  data.LabelField(dtype = torch.float)



fields = [
  ('citing_title', CITING),
  ('cited_title', CITED),
  ('citation_context', CONTEXT),
  ('citation_class_label', LABEL)
]



# load the dataset in json format
train_ds, test_ds = data.TabularDataset.splits(
   path = 'data',
   train = 'SDP_train.csv',
   test = 'SDP_test.csv',
   format = 'csv',
   fields = fields,
   skip_header = True
)



max_input_length = 200





train = pd.read_csv(('data/SDP_train.csv'))