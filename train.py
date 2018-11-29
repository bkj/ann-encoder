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
from helpers import fast_topk, precision_at_ks

from basenet.helpers import set_freeze

# --
# Run

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train-path', type=str)
    parser.add_argument('--test-path', type=str)
    parser.add_argument('--cache-path', type=str, default='./cache/ml20')
    
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
        "train" : list(DataLoader(
            dataset=RaggedAutoencoderDataset(X=X_train, n_toks=n_toks),
            batch_size=args.batch_size,
            collate_fn=ragged_collate_fn,
            num_workers=4,
            pin_memory=True,
            shuffle=True,
        )),
        "valid" : list(DataLoader(
            dataset=RaggedAutoencoderDataset(X=X_train, n_toks=n_toks),
            batch_size=args.batch_size,
            collate_fn=ragged_collate_fn,
            num_workers=4,
            pin_memory=False,
            shuffle=False,
        ))
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
    
    params = [p for p in model.parameters() if p.requires_grad]
    model.init_optimizer(opt=torch.optim.Adam, params=params, lr=args.lr)
    
    # >>
    
    # --
    # 0) Normal training
    
    # {"epoch": 0, "p_at_01": 0.5115926436715214, "p_at_05": 0.40765670467099424, "p_at_10": 0.34342457741546506, "elapsed": 49.7515709400177}
    # {"epoch": 1, "p_at_01": 0.5302000823146297, "p_at_05": 0.42323149906493474, "p_at_10": 0.35684547233434183, "elapsed": 102.56505918502808}
    
    # --
    # 1) Freeze from beginning
    
    # set_freeze(model.linear, True)
    
    # {"epoch": 0, "p_at_01": 0.05284743633252222, "p_at_05": 0.03374755402800141, "p_at_10": 0.02698836764312998, "elapsed": 47.941659927368164}
    # This is terrible
    
    # --
    # 2) Freeze after a short burn-in period
    
    # train_hist = model.train_epoch(dataloaders, mode='train', compute_acc=False, num_batches=100)
    # preds, _   = model.predict(dataloaders, mode='valid', no_cat=True)
    # top_k      = fast_topk(preds, X_train)
    # precisions = precision_at_ks(X_test, top_k)
    # print(json.dumps({
    #     "epoch":   -1,
    #     "p_at_01": precisions[1],
    #     "p_at_05": precisions[5],
    #     "p_at_10": precisions[10],
    #     "elapsed": -1
    # }))
    # set_freeze(model.linear, True)
    
    # {"epoch": -1, "p_at_01": 0.44140137046637734, "p_at_05": 0.34543695349223424, "p_at_10": 0.29227469980432225, "elapsed": -1}
    # {"epoch": 0, "p_at_01": 0.44603698381867674, "p_at_05": 0.3548930270844014, "p_at_10": 0.29877899966063265, "elapsed": 49.190366983413696}
    # All of the learning happens in the burnin, before the last layer is fixed
    
    # --
    # 3) Freeze after first epoch
    
    # ... insert `set_freeze(model.linear, True)` after `model.train_epoch` below
    
    # {"epoch": 0, "p_at_01": 0.5115926436715214, "p_at_05": 0.40765670467099424, "p_at_10": 0.34342457741546506, "elapsed": 49.856390953063965}
    # {"epoch": 1, "p_at_01": 0.49429935086971905, "p_at_05": 0.39147538142722016, "p_at_10": 0.3307806170708989, "elapsed": 101.513028383255}
    # Terrible - Accuracy actually goes down!
    
    # --
    # 4) Freeze rest of model after short burn-in period
    
    # train_hist = model.train_epoch(dataloaders, mode='train', compute_acc=False, num_batches=100)
    # preds, _   = model.predict(dataloaders, mode='valid', no_cat=True)
    # top_k      = fast_topk(preds, X_train)
    # precisions = precision_at_ks(X_test, top_k)
    # print(json.dumps({
    #     "epoch":   -1,
    #     "p_at_01": precisions[1],
    #     "p_at_05": precisions[5],
    #     "p_at_10": precisions[10],
    #     "elapsed": -1
    # }))
    # set_freeze(model.emb, True)
    # set_freeze(model.layers, True)
    
    # {"epoch": -1, "p_at_01": 0.44140137046637734, "p_at_05": 0.34543695349223424, "p_at_10": 0.29227469980432225, "elapsed": -1}
    # {"epoch": 0, "p_at_01": 0.425046753265508, "p_at_05": 0.34323900846974215, "p_at_10": 0.2927093788133696, "elapsed": 46.91078042984009}
    # Accuracy goes down here as well.
    
    # --
    # 5) Freeze everythin but last layer after first epoch
    
    # ... insert
    # ```
    #     set_freeze(model.emb, True)
    #     set_freeze(model.layers, True)
    # ```
    # after `model.train_epoch` below
    
    # {"epoch": 0, "p_at_01": 0.5115926436715214, "p_at_05": 0.40765670467099424, "p_at_10": 0.34342457741546506, "elapsed": 48.89760661125183} 
    # {"epoch": 1, "p_at_01": 0.49590954055439623, "p_at_05": 0.3955405688374142, "p_at_10": 0.33511007776566326, "elapsed": 95.30001187324524}
    # Accuracy goes down here as well.
    
    # --
    # Summary
    
    # These results suggest that the last layer is critical for learning, and we can't just freeze it
    # Next questions:
    #   What if we precompute the softmax normalization?
    
    # <<
    
    # --
    # Run
    
    t = time()
    for epoch in range(args.epochs):
        
        train_hist = model.train_epoch(dataloaders, mode='train', compute_acc=False)
        
        if epoch % args.eval_interval == 0:
            
            # Predict
            preds, _ = model.predict(dataloaders, mode='valid', no_cat=True)
            
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
