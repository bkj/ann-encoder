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

import faiss
import torch
from torch import nn
from torch.nn import functional as F

from basenet import BaseNet
from basenet.helpers import to_numpy, set_seeds

from torch.utils.data import Dataset, DataLoader

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

class ApproxLinear(nn.Module):
    def __init__(self, linear, batch_size, topk, nprobe, npartitions):
        super().__init__()
        
        self.weights = linear.weight.detach().cpu().numpy()
        
        self.cpu_index = faiss.index_factory(
            self.weights.shape[1],
            f"IVF{npartitions},Flat",
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


class EmbeddingSum(nn.Module):
    def __init__(self, n_toks, emb_dim):
        super().__init__()
        
        self.emb = nn.Embedding(n_toks, emb_dim, padding_idx=0)
        self.emb_bias = nn.Parameter(torch.zeros(emb_dim))
        
        self.emb_dim = emb_dim
        
        # torch.nn.init.normal_(self.emb.weight.data, 0, 0.01) # !! Slows down approx. _a lot_ (at high dimensions?)
        # self.emb.weight.data[0] = 0
        
        self._bag = False # !! Faster at inference time, waay slower at training
    
    def set_bag(self, val):
        self._bag = val
        if val:
            self.emb_bag = nn.EmbeddingBag(n_toks, self.emb_dim, mode='sum') 
            self.emb_bag.weight.data.set_(self.emb.weight.data.clone())
            del self.emb
    
    def forward(self, x):
        out = self.emb(x).sum(dim=1) if not self._bag else self.emb_bag(x) 
        out = out + self.emb_bias
        return out


class DestinyLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias_offset=0.0):
        super().__init__(in_channels, out_channels, bias=True) # !! bias not handled by approx yet
        # torch.nn.init.normal_(self.weight.data, 0, 0.01)
        # self.bias.data.zero_()        # !!
        # self.bias.data += bias_offset # !!


class DestinyInferenceModel(BaseNet):
    def __init__(self, n_toks, emb_dim):
        super().__init__()
        
        self.emb = EmbeddingSum(n_toks, emb_dim)
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(0.0),
            
            DestinyLinear(emb_dim, emb_dim, bias_offset=0),
            
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(0.0),
        )
        
        self.linear = DestinyLinear(emb_dim, n_toks)
        
        self.approx_linear = None # Init later
        self.exact         = True
    
    def init_ann(self, topk, batch_size, nprobe, npartitions):
        self.approx_linear = ApproxLinear(self.linear, batch_size, topk, nprobe, npartitions)
    
    def forward(self, x):
        x = self.emb(x)
        x = self.layers(x)
        x = self.linear(x) if self.exact else self.approx_linear(x)
        return x

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
    X_test = [set(x) for x in X_test]
    
    # --
    # Dataloaders
    
    print('define dataloaders', file=sys.stderr)
    dataloaders = {
        "valid" : list(DataLoader( # Precompute w/ `list` call
            dataset=RaggedAutoencoderDataset(X=X_train, n_toks=n_toks),
            batch_size=args.batch_size,
            collate_fn=pad_collate_fn,
            num_workers=2,
            pin_memory=False,
            shuffle=False,
        ))
    }
    
    # --
    # Define model
    
    print('define model', file=sys.stderr)
    model = DestinyInferenceModel(
        n_toks=n_toks,
        emb_dim=args.emb_dim,
    )
    model.load('%s.pt' % args.cache_path)
    model = model.to(torch.device('cuda'))
    model.eval()
    model.verbose = not args.no_verbose
    print(model, file=sys.stderr)
    
    print('set_bag + init_ann', file=sys.stderr)
    model.emb.set_bag(True) # Convert EmbeddingLayer to EmbeddingBag
    model.init_ann(args.topk, args.batch_size, args.nprobe, args.npartitions)
    
    # --
    # Run
    
    torch.cuda.synchronize()
    print('run', file=sys.stderr)
    if args.benchmark:
        
        # Exact
        model.exact = True
        exact_time = benchmark_predict(model, dataloaders, mode='valid')
        
        # Approximate
        model.exact = False
        approx_time = benchmark_predict(model, dataloaders, mode='valid')
        
        print(json.dumps({
            "exact_time"     : exact_time,
            "approx_time"    : approx_time,
            "approx_speedup" : exact_time / approx_time
        }))
    else:
        
        # Exact accuracy
        model.exact = True
        _ = model(warm_batch.cuda())
        torch.cuda.synchronize()
        
        t = time()
        preds, _   = model.predict(dataloaders, mode='valid', no_cat=True)
        infer_time = time() - t
        top_k = fast_topk(preds, X_train)
        
        p_at_01 = np.mean([precision(X_test[i], top_k[i][:1]) for i in range(len(X_test))])
        p_at_05 = np.mean([precision(X_test[i], top_k[i][:5]) for i in range(len(X_test))])
        p_at_10 = np.mean([precision(X_test[i], top_k[i][:10]) for i in range(len(X_test))])
        print(json.dumps({
            "p_at_01"     : p_at_01,
            "p_at_05"     : p_at_05,
            "p_at_10"     : p_at_10,
            "infer_time"  : infer_time,
            "total_time"  : time() - t,
        }))
