#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 09:29:20 2021

@author: amine
"""

import torch
import torch.nn as nn


class Classifier(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout= 0.5):
        
        super().__init__()
        
        self.in_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        
        self.classifier_l1 = nn.Linear(600, hidden_dim)
        self.classifier_l2 = nn.Linear(hidden_dim * input_dim, output_dim)
        
        
    
    def forward(self, input1, input2, input3):
        
        
        input = torch.cat([input1.unsqueeze(0), input2.unsqueeze(0), input3.unsqueeze(0)], 2)
        
        input = input.transpose(0,2)
        
        print("heyyy", input.size())
        
        out = self.classifier_l1(input)
        out = out.view(out.size()[1], out.size()[0] * out.size()[2])
        out = self.classifier_l2(out)
        
        return out
#        return input
        
    
    
class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        
        super().__init__()
        
        self.bert = bert
        
        embedding_dim = bert.config.to_dict()['hidden_size']
        
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [batch size, sent len]
                
        with torch.no_grad():
            embedded = self.bert(text)[0]
                
        #embedded = [batch size, sent len, emb dim]
        
        _, hidden = self.rnn(embedded)
        
        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
                
        #hidden = [batch size, hid dim]
        
        output = self.out(hidden)
        
        #output = [batch size, out dim]
        
        return output