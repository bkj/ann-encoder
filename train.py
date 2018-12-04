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
from torch.utils.data import DataLoader

from basenet.helpers import to_numpy, set_seeds

from model import ExactEncoder
from model import RaggedAutoencoderDataset, ragged_collate_fn
from helpers import fast_topk, precision_at_ks, predict

# --
# Run

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train-path', type=str)
    parser.add_argument('--test-path', type=str)
    parser.add_argument('--cache-path', type=str, required=True)
    
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--emb-dim', type=int, default=800)
    
    parser.add_argument('--epochs', type=int, default=1)
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
        
        X_train = train_edges.groupby(0)[1].apply(lambda x: set(x + 1)).values
        X_test  = test_edges.groupby(0)[1].apply(lambda x: set(x + 1)).values
        
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
    
    # --
    # Dataloaders
    
    print('define dataloaders', file=sys.stderr)
    dataloaders = {
        "train" : DataLoader(
            dataset=RaggedAutoencoderDataset(X=X_train, n_toks=n_toks),
            batch_size=args.batch_size,
            collate_fn=ragged_collate_fn,
            num_workers=4,
            pin_memory=True,
            shuffle=True,
        ),
        "valid" : DataLoader(
            dataset=RaggedAutoencoderDataset(X=X_train, n_toks=n_toks),
            batch_size=args.batch_size,
            collate_fn=ragged_collate_fn,
            num_workers=4,
            pin_memory=False,
            shuffle=False,
        )
    }
    
    # --
    # Model
    
    print('define model', file=sys.stderr)
    model = ExactEncoder(
        n_toks=n_toks,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        bias_offset=args.bias_offset,
    )
    
    model = model.to(torch.device('cuda'))
    model.verbose = not args.no_verbose
    print(model, file=sys.stderr)
    
    model.init_optimizer(opt=torch.optim.Adam, params=model.parameters(), lr=args.lr)
    
    # --
    # Run
    
    t = time()
    for epoch in range(args.epochs):
        train_hist = model.train_epoch(dataloaders, mode='train', compute_acc=False)
        
        if epoch % args.eval_interval == 0:
            
            # Predict
            preds, _ = predict(model, dataloaders, mode='valid', no_cat=True)
            
            # Rank
            top_k = fast_topk(preds, X_train)
            
            # Compute precision
            precisions = precision_at_ks(X_test, top_k)
            
            print(json.dumps({
                "epoch":   epoch,
                "p_at_01": precisions[1],
                "p_at_05": precisions[5],
                "p_at_10": precisions[10],
                "elapsed": time() - t,
            }))
    
    model.save('%s.pt' % args.cache_path)
