#!/usr/bin/env python

"""
    synth.py
    
    Comparison of FAISS lookup and pytorch GPU matmul
    for large softmax output layer
    
    !! Uses the `ann` branch of basenet, where `model.predict` doesn't actually return anything from the GPU
    
    !! Need some way to quantify accuracy
    !! Could also use `IVFx,PQy` index
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from time import time
from collections import OrderedDict

import faiss # !! segfault otherwise?
import torch
from torch import nn

from basenet.helpers import set_seeds

from model import InferenceEncoder

def benchmark_predict(model, dataloaders, mode='val'):
    _ = model.eval()
    for data, _ in dataloaders[mode]:
        with torch.no_grad():
            data = data.cuda(async=True)
            out  = model(data)
    
    torch.cuda.synchronize()


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
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=True
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
    print(model, file=sys.stderr)
    
    model.init_ann(args.topk, args.batch_size, args.nprobe, args.npartitions)
    
    # Warmup
    model.exact = False; _ = model(dataloaders['valid'][0][0].cuda())
    model.exact = True;  _ = model(dataloaders['valid'][0][0].cuda())
    
    # --
    # Run
    
    # Approximate
    t = time()
    model.exact = False
    benchmark_predict(model, dataloaders, mode='valid')
    approx_time = time() - t
    
    # Exact
    t = time()
    model.exact = True
    benchmark_predict(model, dataloaders, mode='valid')
    exact_time = time() - t
    
    print({
        "exact_time"     : exact_time,
        "approx_time"    : approx_time,
        "approx_speedup" : exact_time / approx_time,
    })
