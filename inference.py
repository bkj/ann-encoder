#!/usr/bin/env python

"""
    inference.py
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from joblib import Parallel, delayed

import faiss # !! segfault otherwise?
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from basenet.helpers import to_numpy, set_seeds

from model import ExactEncoder, InferenceEncoder
from model import RaggedAutoencoderDataset, ragged_collate_fn
from helpers import fast_topk, precision_at_ks

# --
# Helpers

def benchmark_predict(model, dataloaders, mode='val'):
    _ = model.eval()
    
    gen = dataloaders[mode]
    if model.verbose:
        gen = tqdm(gen)
    
    t = time()
    for data, _ in gen:
        with torch.no_grad():
            data = data.cuda(async=True)
            out  = model(data)
    
    torch.cuda.synchronize()
    
    return time() - t

def warmup(model, batch):
    batch = batch.cuda()
    
    model.exact = False
    approx_test = model(batch)
    approx_test = to_numpy(approx_test)
    
    model.exact = True
    exact_test  = model(batch)
    exact_test  = exact_test.topk(k=args.topk, dim=-1)[1]
    exact_test  = to_numpy(exact_test)
    
    return (approx_test[:,0] == exact_test[:,0]).mean()


# --
# Run

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cache-path', type=str)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--emb-dim', type=int, default=800)
    
    parser.add_argument('--topk',        type=int, default=32)
    parser.add_argument('--nprobe',      type=int, default=32)
    parser.add_argument('--npartitions', type=int, default=8192)
    
    parser.add_argument('--benchmark', action="store_true")
    
    parser.add_argument('--no-verbose', action="store_true")
    parser.add_argument('--seed', type=int, default=456)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seeds(args.seed)
    
    # --
    # IO
    
    print('loading cache: start', file=sys.stderr)
    X_train = np.load('%s_train.npy' % args.cache_path)
    X_test  = np.load('%s_test.npy' % args.cache_path)
    print('loading cache: done', file=sys.stderr)
    
    n_toks = max([max(x) for x in X_train]) + 1
    
    # --
    # Dataloaders
    
    print('define dataloaders', file=sys.stderr)
    dataloaders = {
        "valid" : list(DataLoader(
            dataset=RaggedAutoencoderDataset(X=X_train, n_toks=n_toks),
            collate_fn=ragged_collate_fn,
            batch_size=args.batch_size,
            pin_memory=True,
            shuffle=False,
        ))
    }
    
    # --
    # Define model
    
    print('define model', file=sys.stderr)
    model = InferenceEncoder(
        n_toks=n_toks,
        emb_dim=args.emb_dim,
    )
    
    model.load('%s.pt' % args.cache_path)
    _ = model.eval()
    
    model = model.to(torch.device('cuda'))
    model.verbose = not args.no_verbose
    print(model, file=sys.stderr)
    
    model.init_ann(args.topk, args.batch_size, args.nprobe, args.npartitions)
    
    pct_agree = warmup(model, dataloaders['valid'][0][0])
    print('warmup: pct_agree=%f' % pct_agree, file=sys.stderr)
    
    # --
    # Run
    
    if args.benchmark:
        
        # Approximate
        model.exact = False
        approx_time = benchmark_predict(model, dataloaders, mode='valid')
        
        # Exact
        model.exact = True
        exact_time  = benchmark_predict(model, dataloaders, mode='valid')
        
        print(json.dumps({
            "exact_time"     : exact_time,
            "approx_time"    : approx_time,
            "approx_speedup" : exact_time / approx_time,
        }))
        
    else:
        
        # Exact accuracy
        model.exact = True
        
        preds, _         = model.predict(dataloaders, mode='valid', no_cat=True)
        top_k            = fast_topk(preds, X_train)
        exact_precisions = precision_at_ks(X_test, top_k)
        
        # # Approx accuracy
        # model.exact = False
        # model.approx_linear.dense = True
        
        # preds, _          = model.predict(dataloaders, mode='valid', no_cat=True)
        # top_k             = fast_topk(preds, X_train)
        # approx_precisions = precision_at_ks(X_test, top_k)
        
        print(json.dumps({
            "exact_p_at_01"  : exact_precisions[1],
            "exact_p_at_05"  : exact_precisions[5],
            "exact_p_at_10"  : exact_precisions[10],
            # "approx_p_at_01" : approx_precisions[1],
            # "approx_p_at_05" : approx_precisions[5],
            # "approx_p_at_10" : approx_precisions[10],
        }))
        

