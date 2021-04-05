#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 14:03:38 2021

@author: amine
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, AutoTokenizer, BertModel, AutoModel
from torchtext import data
from models import BERTGRUSentiment, Classifier





#Sentences we want sentence embeddings for
sentences = ['This framework generates embeddings for each input sentence',
             'Sentences are passed as a list of string.',
             'The quick brown fox jumps over the lazy dog.']

#Load AutoModel from huggingface model repository




def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc


def train(encoder_1, encoder_2, encoder_3, model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        print('ok')
        optimizer.zero_grad()
        
        in1 = encoder_1(batch.citing_title)
        in2 = encoder_2(batch.cited_title)
        in3 = encoder_3(batch.citation_context)
        
        print(in1.size(), 'heyyy')
        
        
        predictions = model(in1, in2, in3).squeeze(1)
        
        loss = criterion(predictions, batch.citation_class_label)
        
        acc = binary_accuracy(predictions, batch.citation_class_label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


max_input_length = 200

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained("bert_models/ce_roberta_base", local_files_only = True)

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


LABEL =  data.LabelField(dtype = torch.float, use_vocab = True)



fields = [
  ('citing_title', CITING),
  ('cited_title', CITED),
  ('citation_context', CONTEXT),
  ('citation_class_label', LABEL)
]





# load the dataset in json format
train_ds, valid_ds, test_ds = data.TabularDataset.splits(
   path = '../data',
   train = 'train.csv',
   validation = 'dev.csv',
   test = 'test.csv',
   format = 'csv',
   fields = fields,
   skip_header = True
)


# CITING.build_vocab(train_ds)
# CITED.build_vocab(train_ds)
# CONTEXT.build_vocab(train_ds)
LABEL.build_vocab(train_ds)


train_it, valid_it, test_it = data.BucketIterator.splits(
  (train_ds, valid_ds, test_ds),
  sort_key = lambda x: x.cited_title,
  sort = True,
  batch_size = 2,
  device = 'cpu'
)


bert = BertModel.from_pretrained('bert-base-uncased')

HIDDEN_DIM = 256
ENCODER_OUT = 200
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

encoder_1 =  AutoModel.from_pretrained("bert_models/ce_roberta_base", local_files_only = True)

encoder_2 =  AutoModel.from_pretrained("bert_models/ce_roberta_base", local_files_only = True)

encoder_3 =  AutoModel.from_pretrained("bert_models/ce_roberta_base", local_files_only = True)

model = Classifier(768 * 3, HIDDEN_DIM, OUTPUT_DIM, DROPOUT)


import torch.optim as optim

optimizer = optim.Adam(model.parameters())

criterion = nn.BCEWithLogitsLoss()

device = 'cpu'
model = model.to(device)
criterion = criterion.to(device)


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

epoch_loss = 0
epoch_acc = 0

model.train()
for batch in train_it:
    print('ok')
    optimizer.zero_grad()
    print(batch.citing_title.size())
    in1 = encoder_1(batch.citing_title)[0][:,-1,:]
    in2 = encoder_2(batch.cited_title)[0][:,-1,:]
    in3 = encoder_3(batch.citation_context)[0][:,-1,:]
    
    
    
    print(torch.squeeze(in1, 0).size())
    print(in1.size())
        
        
    predictions = model(in1, in2, in3, pad_token_idx).squeeze(1)
        
    loss = criterion(predictions, batch.citation_class_label)
    
    acc = binary_accuracy(predictions, batch.citation_class_label)
    
    loss.backward()
    
    optimizer.step()
    
    epoch_loss += loss.item()
    epoch_acc += acc.item()
    break

x = train(encoder_1, encoder_2, encoder_3, model, train_it, optimizer, criterion)
        
        
        


