#!/usr/bin/env python

"""
    prep.py
"""

from __future__ import print_function, division

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='data/ml-20m/ratings.csv')
    parser.add_argument('--outpath', type=str, default='data/ml-20m')
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    
    print('loading %s' % args.inpath, file=sys.stderr)
    edges = pd.read_csv(args.inpath)
    del edges['timestamp']
    
    train, test = train_test_split(edges, train_size=0.8, stratify=edges.userId)
    
    train = train.sort_values(['userId', 'movieId']).reset_index(drop=True)
    train['movieRating'] = train.movieId.astype(str)# + ',' + train.rating.astype(str)
    train_feats = train.groupby('userId').movieRating.apply(lambda x: ':'.join(x))
    
    test  = test.sort_values(['userId', 'movieId']).reset_index(drop=True)
    test['movieRating'] = test.movieId.astype(str)# + ',' + test.rating.astype(str)
    test_feats  = test.groupby('userId').movieRating.apply(lambda x: ':'.join(x))
    
    assert train_feats.shape[0] == test_feats.shape[0]
    
    train_feats.to_csv(os.path.join(args.outpath, 'train.txt'), sep='\t', header=None)
    test_feats.to_csv(os.path.join(args.outpath, 'test.txt'), sep='\t', header=None)