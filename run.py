import time
import pandas as pd
import numpy as np
import pickle
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import TabularDataset, Field, NestedField, Iterator

from utils import *
from model import Unified_xtransformerXd, xtransformer1d, xtransformer2d, Unified_xtransformer1d

# torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import warnings
warnings.filterwarnings("ignore")

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def contrib_itos(itos, token_ids, contrib):
    # itos: id to string mapping
    # token_ids: (seq_ln, token_cnt) or (seq_ln,)
    # contrib: (seq_ln, token_cnt, nclass) or (seq_ln, nclass)
    # return:
    # ---> tokens: a list of list of tokens
    # ---> cb: a list of list of contributions (numeric)
    
    tokens = []
    cb = []
    if len(token_ids.shape) == 2:
        for r1,r2 in zip(token_ids, contrib):
            tk = [itos[u] for u in r1 if itos[u] != '<pad>']
            tokens.append(tk)
            cb.append(r2[:len(tk)])
    if len(token_ids.shape) == 1:
        tokens = [itos[u] for u in token_ids if itos[u] != '<pad>']
        cb = contrib[:len(tokens)]
    return tokens, cb

def train(model, iterator, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for _, batch in enumerate(iterator):
        
        inputs = (batch.ego_netloc.squeeze(-1),
                  batch.ngh_netloc.squeeze(-1))
        label = batch.label
        
        optimizer.zero_grad()
        for tensor in inputs:
            tensor.to(device)
        
        output = model(inputs)
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            
            inputs = (batch.ego_netloc.squeeze(-1),
                      batch.ngh_netloc.squeeze(-1))
            label = batch.label
            
            for tensor in inputs:
                tensor.to(device)
            
            output = model(inputs)
            loss = criterion(output,label)
            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)

def test(model, iterator, device):
    
    model.eval()
    
    output_lst, target_lst = [], []
    
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            
            inputs = (batch.ego_netloc.squeeze(-1),
                      batch.ngh_netloc.squeeze(-1))
            target = batch.label
            
            for tensor in inputs:
                tensor.to(device)
            
            output = model(inputs)
            
            output_lst.append(output)
            target_lst.append(target)
    
    return output_lst, target_lst




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='data.json', help='which data to use')
    parser.add_argument('--verbose', type=str, default='both', help='which sequence(s) to use')
    args = parser.parse_args()
    
    BATCH_SIZE = 16
    nclass = 2
    emb_sz = 64
    
    dim_feedforward = 1024
    nhead = 4
    nlayer = 2
    
    dropout = 0.8
    
    
    ## === load data === ##
    NETLOC = NestedField(Field())
    LABEL = Field(sequential=False, use_vocab=False)
    WEEK = Field(sequential=False, use_vocab=False)

    fields = {"ego_netloc": ("ego_netloc", NETLOC), # use same field for two columns
              "ngh_netloc": ("ngh_netloc", NETLOC),
              "week": ("week", WEEK),
              "label": ("label",LABEL)}
    data = TabularDataset(path='../dataset/{}'.format(args.filename), format='json',fields=fields)
    NETLOC.build_vocab(data.ego_netloc, data.ngh_netloc, min_freq=10)
    ntoken = len(NETLOC.vocab)
    
    trn,tst,val = data.split([0.6,0.2,0.2], stratified=True)
    
    
    # divide into buckets
    trn_iter = Iterator(trn, batch_size=BATCH_SIZE, shuffle=True, device=device)
    val_iter = Iterator(val, batch_size=BATCH_SIZE, shuffle=True, device=device)
    tst_iter = Iterator(tst, batch_size=BATCH_SIZE, shuffle=False, device=device)
    
    ## === setting environment === ##
    criterion = nn.CrossEntropyLoss() # combination of LogSoftmax and NLLoss
    # model = Unified_xtransformerXd(ntokens, dim_feedforwards, nheads, nlayers, dropout, emb_sz, nclass).to(device)
    model = Unified_xtransformer1d(ntoken, dim_feedforward, nhead, nlayer, dropout, emb_sz, nclass, args.verbose).to(device)
    # model = xtransformer1d(ntoken, emb_sz, dim_feedforward, nhead, nlayer, nclass, dropout)
    model.apply(init_weights)
    
    if torch.cuda.device_count() > 1:
        print("let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001) # 1e-3 is good, 1e-4 is too small/slow
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    
    ## === run model == ##
    N_EPOCHS = 50

    PATH = 'ckpt/best_model'

    best_valid_loss = float('inf')
    no_improvement = 0

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, trn_iter, optimizer, criterion, device)
        valid_loss = evaluate(model, val_iter, criterion, device)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print('Epoch: %02d | Time: %02dm %02ds | Train Loss: %.3f | Eval Loss: %.3f' % (
            epoch, epoch_mins, epoch_secs, train_loss, valid_loss))

        if valid_loss > best_valid_loss:
            no_improvement += 1
        elif valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), PATH)
            no_improvement = 0
        if no_improvement == 1:
            break

    ## === test model === ##
    model.load_state_dict(torch.load('ckpt/best_model'))
    
    
    output_lst, target_lst = test(model, tst_iter, device)
    output = torch.cat(output_lst, 0)
    target = torch.cat(target_lst, 0)
    
    acc,auc,pre,rec,f1 = perf(output,target)
    
    print('filename: %s, verbose: %s' % (args.filename, args.verbose))
    
    print('acc: %.3f' % acc)
    print('auc: %.3f' % auc)
    print('pre: %.3f' % pre)
    print('rec: %.3f' % rec)
    print('f1 : %.3f' % f1)
    
