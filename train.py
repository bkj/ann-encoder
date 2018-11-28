#!/usr/bin/env python

"""
    train.py
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from time import time
from joblib import Parallel, delayed

import torch
from torch import nn
from torch.nn import functional as F

from basenet import BaseNet, HPSchedule
from basenet.helpers import to_numpy, set_seeds

from torch.utils.data import Dataset, DataLoader

# --
# Helpers

def precision(act, preds):
    return len(act.intersection(preds)) / preds.shape[0]

def __filter_and_rank(pred, X_filter, k=10):
    for i in range(pred.shape[0]):
        pred[i][X_filter[i]] = -1
    
    return np.argsort(-pred, axis=-1)[:,:k]

def fast_topk(preds, X_train, n_jobs=32):
    offsets = np.cumsum([p.shape[0] for p in preds])
    offsets -= preds[0].shape[0]
    
    jobs = [delayed(__filter_and_rank)(
        to_numpy(pred),
        to_numpy(X_train[offset:(offset + pred.shape[0])])
    ) for pred, offset in zip(preds, offsets)]
    top_k = Parallel(n_jobs=n_jobs, backend='threading')(jobs)
    top_k = np.vstack(top_k)
    
    return top_k

# --
# Data utilities

class RaggedAutoencoderDataset(Dataset):
    def __init__(self, X, n_toks):
        self.X = [torch.LongTensor(xx) for xx in X]
        self.n_toks = n_toks
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = torch.zeros((self.n_toks,))
        y[x] += 1
        return self.X[idx], y
    
    def __len__(self):
        return len(self.X)

def pad_collate_fn(batch, pad_value=0):
    X, y = zip(*batch)
    
    max_len = max([len(xx) for xx in X])
    X = [F.pad(xx, pad=(max_len - len(xx), 0), value=pad_value).data for xx in X]
    
    X = torch.stack(X, dim=-1).t().contiguous()
    y = torch.stack(y, dim=0)
    return X, y

# --
# Model definition

class EmbeddingSum(nn.Module):
    def __init__(self, n_toks, emb_dim):
        super().__init__()
        self.emb     = nn.Embedding(n_toks, emb_dim, padding_idx=0)
        self.emb_bias = nn.Parameter(torch.zeros(emb_dim))
        
        torch.nn.init.normal_(self.emb.weight.data, 0, 0.01) # !! Slows down approx. _a lot_ (at high dimensions?)
        self.emb.weight.data[0] = 0
        
    def forward(self, x):
        out = self.emb(x).sum(dim=1)
        out = out + self.emb_bias
        return out


class DestinyLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias_offset=0.0):
        super().__init__(in_channels, out_channels, bias=True) # !! bias not handled by approx yet
        torch.nn.init.normal_(self.weight.data, 0, 0.01)
        self.bias.data.zero_()        # !!
        self.bias.data += bias_offset # !!


class DestinyModel(BaseNet):
    def __init__(self, n_toks, emb_dim, dropout, bias_offset):
        super().__init__(loss_fn=F.binary_cross_entropy_with_logits)
        
        self.emb = EmbeddingSum(n_toks, emb_dim)
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(dropout),
            
            DestinyLinear(emb_dim, emb_dim, bias_offset=0),
            
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(dropout),
        )
        
        self.linear = DestinyLinear(emb_dim, n_toks, bias_offset=bias_offset)
        
    def forward(self, x):
        x = self.emb(x)
        x = self.layers(x)
        x = self.linear(x)
        return x

# --
# Run

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train-path', type=str)
    parser.add_argument('--test-path', type=str)
    parser.add_argument('--cache-path', type=str)
    
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--emb-dim', type=int, default=800)
    
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--bias-offset', type=float, default=-10)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    parser.add_argument('--eval-interval', type=int, default=1)
    parser.add_argument('--no-verbose', action="store_true")
    parser.add_argument('--seed', type=int, default=456)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seeds(args.seed)
    
    # --
    # IO
    
    if args.train_path is not None:
        assert args.test_path is not None
        
        train_edges = pd.read_csv(args.train_path, header=None, sep='\t')[[0,1]]
        test_edges  = pd.read_csv(args.test_path, header=None, sep='\t')[[0,1]]
        
        X_train = train_edges.groupby(0)[1].apply(lambda x: list(x + 1)).values # Increment by 1 for padding_idx
        X_test  = test_edges.groupby(0)[1].apply(lambda x: list(x + 1)).values  # Increment by 1 for padding_idx
        
        o = np.argsort([len(t) for t in X_test])[::-1]
        X_train, X_test = X_train[o], X_test[o]
        
        np.save('%s_train.npy' % args.cache_path, X_train)
        np.save('%s_test.npy' % args.cache_path, X_test)
    else:
        print('loading cache: start', file=sys.stderr)
        X_train = np.load('%s_train.npy' % args.cache_path)
        X_test  = np.load('%s_test.npy' % args.cache_path)
        print('loading cache: done', file=sys.stderr)
    
    n_toks = max([max(x) for x in X_train]) + 1
    X_test = [set(x) for x in X_test]
    
    # --
    # Dataloaders
    
    print('define dataloaders', file=sys.stderr)
    dataloaders = {
        "train" : DataLoader(
            dataset=RaggedAutoencoderDataset(X=X_train, n_toks=n_toks),
            batch_size=args.batch_size,
            collate_fn=pad_collate_fn,
            num_workers=2,
            pin_memory=True,
            shuffle=True,
        ),
        "valid" : DataLoader(
            dataset=RaggedAutoencoderDataset(X=X_train, n_toks=n_toks),
            batch_size=args.batch_size,
            collate_fn=pad_collate_fn,
            num_workers=2,
            pin_memory=False,
            shuffle=False,
        )
    }
    
    print('define model', file=sys.stderr)
    model = DestinyModel(
        n_toks=n_toks,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        bias_offset=args.bias_offset,
    ).to(torch.device('cuda'))
    model.verbose = not args.no_verbose
    print(model, file=sys.stderr)
    
    model.init_optimizer(opt=torch.optim.Adam, params=model.parameters(), lr=args.lr)
    
    # Could make validation run faster
    # print('preloading dataloaders["valid"] + warming up', file=sys.stderr)
    # dataloaders['valid'] = list(dataloaders['valid'])
    
    _ = model(next(iter(dataloaders['valid']))[0].cuda())
    
    t = time()
    for epoch in range(args.epochs):
        train_hist = model.train_epoch(dataloaders, mode='train', compute_acc=False)
        
        if epoch % args.eval_interval == 0:
            preds, _ = model.predict(dataloaders, mode='valid', no_cat=True)
            top_k = fast_topk(preds, X_train)
            
            p_at_01 = np.mean([precision(X_test[i], top_k[i][:1]) for i in range(len(X_test))])
            p_at_05 = np.mean([precision(X_test[i], top_k[i][:5]) for i in range(len(X_test))])
            p_at_10 = np.mean([precision(X_test[i], top_k[i][:10]) for i in range(len(X_test))])
            print(json.dumps({
                "epoch":   epoch,
                "p_at_01": p_at_01,
                "p_at_05": p_at_05,
                "p_at_10": p_at_10,
                "elapsed": time() - t,
            }))
    
    model.save('%s.pt' % args.cache_path)
