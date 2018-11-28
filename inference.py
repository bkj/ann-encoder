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

from model import InferenceEncoder
from model import RaggedAutoencoderDataset, ragged_collate_fn
from helpers import fast_topk, precision_at_ks

# --
# Helpers

def benchmark_predict(model, dataloaders, mode='val'):
    _ = model.eval()
    
    _ = model(dataloaders[mode][0][0].cuda())
    torch.cuda.synchronize()
    
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
        "valid" : list(DataLoader( # Precompute w/ `list` call
            dataset=RaggedAutoencoderDataset(X=X_train, n_toks=n_toks),
            batch_size=args.batch_size,
            collate_fn=ragged_collate_fn,
            num_workers=4,
            pin_memory=False,
            shuffle=False,
        ))
    }
    
    # --
    # Define model
    
    print('define model', file=sys.stderr)
    model = InferenceEncoder(n_toks=n_toks, emb_dim=args.emb_dim)
    model.load('%s.pt' % args.cache_path)
    model = model.to(torch.device('cuda'))
    model.eval()
    model.verbose = not args.no_verbose
    print(model, file=sys.stderr)
    
    if args.benchmark:
        print('set_bag + init_ann', file=sys.stderr)
        model.init_ann(args.topk, args.batch_size, args.nprobe, args.npartitions)
    
    # --
    # Run
    
    torch.cuda.synchronize()
    print('run', file=sys.stderr)
    if args.benchmark:
        
        # Exact
        model.exact = True
        exact_time  = benchmark_predict(model, dataloaders, mode='valid')
        
        # Approximate
        model.exact = False
        approx_time = benchmark_predict(model, dataloaders, mode='valid')
        
        print(json.dumps({
            "exact_time"     : exact_time,
            "approx_time"    : approx_time,
            "approx_speedup" : exact_time / approx_time
        }))
    else:
        t = time()
        
        # Exact accuracy
        model.exact = True
        
        # Predict
        preds, _ = model.predict(dataloaders, mode='valid', no_cat=True)
        
        # Rank
        top_k = fast_topk(preds, X_train)
        
        # Compute precision
        precisions = precision_at_ks(X_test, top_k)
        
        print(json.dumps({
            "p_at_01": precisions[1],
            "p_at_05": precisions[5],
            "p_at_10": precisions[10],
            "elapsed": time() - t,
        }))
