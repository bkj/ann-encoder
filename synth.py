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
from joblib import Parallel, delayed

import faiss
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from basenet import BaseNet
from basenet.helpers import set_seeds

from torch.utils.data import Dataset, DataLoader

def benchmark_predict(model, dataloaders, mode='val', no_cat=False, dummy=False):
    _ = model.eval()
    for data, _ in dataloaders[mode]:
        with torch.no_grad():
            data = data.cuda(async=True)
            out  = model(data)
    
    torch.cuda.synchronize()


class ApproxLinear(nn.Module):
    def __init__(self, linear, batch_size, topk, nprobe, npartitions):
        super().__init__()
        
        self.weights = linear.weight.detach().cpu().numpy()
        
        self.cpu_index = faiss.index_factory(
            self.weights.shape[1],
            f"IVF{npartitions},Flat",
            # "Flat",
            
            faiss.METRIC_INNER_PRODUCT # This appears to be slower -- why? And can we get away w/ L2 at inference time?
        )
        self.cpu_index.train(self.weights)
        self.cpu_index.add(self.weights)
        self.cpu_index.nprobe = nprobe
        
        self.res   = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(self.res, 0, self.cpu_index)
        
        self.topk = topk
        self.I    = torch.LongTensor(batch_size, topk).cuda()
        self.D    = torch.FloatTensor(batch_size, topk).cuda()
        self.Dptr = faiss.cast_integer_to_float_ptr(self.D.storage().data_ptr())
        self.Iptr = faiss.cast_integer_to_long_ptr(self.I.storage().data_ptr())
        
    def forward(self, x):
        xptr = faiss.cast_integer_to_float_ptr(x.storage().data_ptr())
        self.index.search_c(
            x.shape[0],
            xptr,
            self.topk,
            self.Dptr,
            self.Iptr,
        )
        return self.I


class Model(BaseNet):
    def __init__(self, n_toks, emb_dim, out_dim, batch_size, topk, nprobe, npartitions):
        super().__init__()
        
        self.emb           = nn.EmbeddingBag(n_toks, emb_dim)
        self.linear        = nn.Linear(emb_dim, out_dim, bias=False)
        self.approx_linear = ApproxLinear(self.linear, batch_size, topk, nprobe, npartitions)
        
        self.topk = topk
        self.exact = None
    
    def forward(self, x):
        assert self.exact is not None
        
        x = self.emb(x)
        
        if self.exact:
            x = self.linear(x)
            # x = x.topk(k=self.topk, dim=-1)[1] # Expensive!
        else:
            x = self.approx_linear(x)
        
        return x


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
    # Model
    
    print('define model', file=sys.stderr)
    model = Model(
        n_toks=args.n_toks,
        emb_dim=args.emb_dim,
        out_dim=args.out_dim,
        batch_size=args.batch_size,
        topk=args.topk,
        nprobe=args.nprobe,
        npartitions=args.npartitions,
    ).to(torch.device('cuda'))
    model.verbose = args.verbose
    
    # Warmup
    model.exact = True;  _ = model(dataloaders['valid'][0][0].cuda())
    model.exact = False; _ = model(dataloaders['valid'][0][0].cuda())
    
    # --
    # Run
    
    # Approximate
    t = time()
    model.exact = False
    benchmark_predict(model, dataloaders, mode='valid', dummy=True)
    approx_time = time() - t
    
    # Exact
    t = time()
    model.exact = True
    benchmark_predict(model, dataloaders, mode='valid', dummy=True)
    exact_time = time() - t
    
    print({
        "exact_time"     : exact_time,
        "approx_time"    : approx_time,
        "approx_speedup" : exact_time / approx_time,
    })
