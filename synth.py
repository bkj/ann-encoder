#!/usr/bin/env python

"""
    synth.py
    
    Comparison of FAISS lookup and pytorch GPU matmul
    for large softmax output layer
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from collections import OrderedDict

import faiss # !! segfault otherwise?
import torch
from torch import nn

from basenet.helpers import to_numpy, set_seeds

from model import InferenceEncoder
from helpers import benchmark_predict

# --
# Helpers

def warmup(model, batch, topk):
    batch = batch.cuda()
    
    model.exact = False
    approx_test = model(batch)
    approx_test = to_numpy(approx_test)
    
    model.exact = True
    exact_test  = model(batch)
    exact_test  = exact_test.topk(k=topk, dim=-1)[1]
    exact_test  = to_numpy(exact_test)
    
    return (approx_test[:,0] == exact_test[:,0]).mean()

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n-toks',      type=int, default=25878) # only effects embedding layer overhead
    parser.add_argument('--seq-len',     type=int, default=100)   # only effects embedding layer overhead
    parser.add_argument('--n-batches',   type=int, default=100)   # only effects embedding layer overhead
    
    parser.add_argument('--batch-size',  type=int, default=2048)
    parser.add_argument('--emb-dim',     type=int, default=128)
    
    parser.add_argument('--out-dim',     type=int, default=400000)
    parser.add_argument('--topk',        type=int, default=32)
    parser.add_argument('--nprobe',      type=int, default=32)
    parser.add_argument('--npartitions', type=int, default=8192)
    
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--verbose', action="store_true")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seeds(args.seed)
    
    # --
    # Generate some data
    
    print('define dataloaders', file=sys.stderr)
    X = np.random.choice(args.n_toks, (args.n_batches * args.batch_size, args.seq_len))
    X = torch.LongTensor(X)
    
    y = torch.zeros(X.shape[0])
    dataloaders = {
        "valid" : list(torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(X, y),
            batch_size=args.batch_size,
            pin_memory=True,
            shuffle=False,
        ))
    }
    
    # --
    # Define model
    
    print('define model', file=sys.stderr)
    model = InferenceEncoder(
        n_toks=args.n_toks,
        emb_dim=args.emb_dim,
        out_dim=args.out_dim,
    )
    
    model = model.to(torch.device('cuda'))
    model.verbose = args.verbose
    _ = model.eval()
    print(model, file=sys.stderr)
    
    model.init_ann(args.topk, args.batch_size, args.nprobe, args.npartitions)
    
    pct_agree = warmup(model=model, batch=dataloaders['valid'][0][0], topk=args.topk)
    print('warmup: pct_agree=%f' % pct_agree, file=sys.stderr)
    
    # --
    # Run
    
    # Approximate
    model.exact = False
    approx_time = benchmark_predict(model, dataloaders, mode='valid')
    
    # Exact
    model.exact = True
    exact_time = benchmark_predict(model, dataloaders, mode='valid')
    
    print({
        "exact_time"     : exact_time,
        "approx_time"    : approx_time,
        "approx_speedup" : exact_time / approx_time,
    })
