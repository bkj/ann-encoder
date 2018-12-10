#!/usr/bin/env python

"""
    model.py
"""

import sys
import numpy as np

import faiss
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from basenet import BaseNet

# --
# Data utilities

class RaggedAutoencoderDataset(Dataset):
    def __init__(self, X, n_toks, maxlen=None):
        if maxlen is None:
            self.X = [torch.LongTensor(xx) for xx in X]
        else:
            self.X = [torch.LongTensor(xx)[:maxlen] for xx in X]
        
        self.n_toks = n_toks
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = torch.zeros((self.n_toks,))
        y[x] += 1
        return self.X[idx], y
    
    def __len__(self):
        return len(self.X)

def ragged_collate_fn(batch, pad_value=0):
    X, y = zip(*batch)
    
    max_len = max([len(xx) for xx in X])
    X = [F.pad(xx, pad=(max_len - len(xx), 0), value=pad_value).data for xx in X]
    
    X = torch.stack(X, dim=-1).t().contiguous()
    y = torch.stack(y, dim=0)
    return X, y

# --
# Train time models

class BiasedEmbeddingSum(nn.Module):
    def __init__(self, n_toks, emb_dim):
        super().__init__()
        self.emb      = nn.Embedding(n_toks, emb_dim, padding_idx=0)
        self.emb_bias = nn.Parameter(torch.zeros(emb_dim))
        
        torch.nn.init.normal_(self.emb.weight.data, 0, 0.01)
        self.emb.weight.data[0] = 0
        
    def forward(self, x):
        return self.emb(x).sum(dim=1) + self.emb_bias


class BiasedLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias_offset=0.0):
        super().__init__(in_channels, out_channels, bias=True)
        torch.nn.init.normal_(self.weight.data, 0, 0.01)
        self.bias.data.zero_()
        self.bias.data += bias_offset


class ExactEncoder(BaseNet):
    def __init__(self, n_toks, emb_dim, dropout, bias_offset):
        super().__init__(loss_fn=F.binary_cross_entropy_with_logits)
        
        self.emb = BiasedEmbeddingSum(n_toks, emb_dim)
        self.layers = nn.Sequential(*[
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(dropout),
            
            BiasedLinear(emb_dim, emb_dim, bias_offset=0),
            
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(dropout),
            
            BiasedLinear(emb_dim, emb_dim, bias_offset=0),
            
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(dropout),
        ])
        
        self.linear = BiasedLinear(emb_dim, n_toks, bias_offset=bias_offset)
        
    def forward(self, x):
        x = self.emb(x)
        x = self.layers(x)
        x = self.linear(x)
        return x

# --
# Inference time models

class ApproxLinear(nn.Module):
    def __init__(self, linear, batch_size, topk, nprobe, npartitions, flat):
        super().__init__()
        
        linear_weight = linear.weight.detach().cpu().numpy()
        
        self.has_bias = linear.bias is not None
        if self.has_bias:
            linear_bias = linear.bias.detach().cpu().numpy()
            self.weights = np.column_stack([linear_weight, linear_bias])
        else:
            self.weights = linear_weight
        
        if flat:
            print("!! ApproxLinear: FLAT testing mode", file=sys.stderr)
            
        self.cpu_index = faiss.index_factory(
            self.weights.shape[1],
            "Flat" if flat else f"IVF{npartitions},Flat",
            faiss.METRIC_INNER_PRODUCT
        )
        self.cpu_index.train(self.weights)
        self.cpu_index.add(self.weights)
        self.cpu_index.nprobe = nprobe
        
        self.res   = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(self.res, 0, self.cpu_index)
        
        self.topk       = topk
        self.batch_size = batch_size
        self.ones       = torch.ones(batch_size).view(-1, 1).cuda()
        
        self.I    = torch.LongTensor(batch_size, topk).cuda()
        self.D    = torch.FloatTensor(batch_size, topk).cuda()
        self.Dptr = faiss.cast_integer_to_float_ptr(self.D.storage().data_ptr())
        self.Iptr = faiss.cast_integer_to_long_ptr(self.I.storage().data_ptr())
        
        self.dense   = False
        self.out_dim = linear_weight.shape[0]
    
    def _compute_dense(self, I, D):
        col = I.cpu().view(-1)
        row = torch.arange(I.shape[0]).view(-1, 1).repeat(1, I.shape[1]).view(-1).long()
        val = D.cpu().view(-1)
        
        dense = torch.zeros(I.shape[0], self.out_dim) + D.min().cpu() - 1
        dense[(row, col)] = val
        
        return dense
    
    def forward(self, x):
        self.D.zero_()
        self.I.zero_()
        torch.cuda.synchronize() # !! faiss and pytorch on different streams
        
        # !! bs == I.shape[0] most of the time -- do the `:bs` calls slow us down?
        bs = x.shape[0]
        if self.has_bias:
            x = torch.cat([x, self.ones[:bs]], dim=-1)
        
        xptr = faiss.cast_integer_to_float_ptr(x.storage().data_ptr())
        self.index.search_c(
            bs,
            xptr,
            self.topk,
            self.Dptr,
            self.Iptr,
        )
        
        torch.cuda.synchronize()
        self.res.syncDefaultStreamCurrentDevice()
        
        if self.dense:
            return self._compute_dense(self.I[:bs], self.D[:bs])
        else:
            return self.I[:bs]


class InfBiasedEmbeddingSum(nn.Module):
    def __init__(self, n_toks, emb_dim):
        super().__init__()
        
        self.emb = nn.EmbeddingBag(n_toks, emb_dim, mode='sum')
        self.emb_bias = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        return self.emb(x) + self.emb_bias


class InfBiasedLinear(nn.Linear):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, bias=True)


class InferenceEncoder(BaseNet):
    def __init__(self, n_toks, emb_dim, out_dim=None):
        super().__init__()
        
        self.n_toks = n_toks
        
        if out_dim is None:
            out_dim = n_toks
        
        self.emb = InfBiasedEmbeddingSum(n_toks, emb_dim)
        self.layers = nn.Sequential(*[
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(0.0),
            
            InfBiasedLinear(emb_dim, emb_dim),
            
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(0.0),
        ])
        
        self.linear = InfBiasedLinear(emb_dim, out_dim)
        
        self.approx_linear = None
        self.exact         = True
    
    def init_ann(self, topk, batch_size, nprobe, npartitions, flat=False):
        self.approx_linear = ApproxLinear(
            self.linear, batch_size, topk, nprobe, npartitions, flat=flat)
    
    def forward(self, x):
        x = self.emb(x)
        x = self.layers(x)
        x = self.linear(x) if self.exact else self.approx_linear(x)
        return x
