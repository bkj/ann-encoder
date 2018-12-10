#!/usr/bin/env python

"""
    utils/prep_ml.py
    
    Data available at: http://files.grouplens.org/datasets/movielens/ml-20m.zip
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
    parser.add_argument('--inpath', type=str, default='data/ml-1m/ratings.csv')
    parser.add_argument('--outpath', type=str, default='data/ml-1m')
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    
    print('loading -> %s' % args.inpath, file=sys.stderr)
    edges = pd.read_csv(args.inpath)
# >>
    edges = edges.sort_values('timestamp').reset_index(drop=True)
    
    train_sel = edges.userId.duplicated(keep='last')
    train, test = edges[train_sel], edges[~train_sel]
# >>
    del train['timestamp']
    del test['timestamp']
    
    train = train.sort_values(['userId', 'movieId']).reset_index(drop=True)
    # train['movieRating'] = train.movieId.astype(str)# + ',' + train.rating.astype(str)
    # train_feats = train.groupby('userId').movieRating.apply(lambda x: ':'.join(x))
    
    test  = test.sort_values(['userId', 'movieId']).reset_index(drop=True)
    # test['movieRating'] = test.movieId.astype(str)# + ',' + test.rating.astype(str)
    # test_feats  = test.groupby('userId').movieRating.apply(lambda x: ':'.join(x))
    
    # assert train_feats.shape[0] == test_feats.shape[0]
    
    print('saving -> %s' % args.outpath, file=sys.stderr)
    train.to_csv(os.path.join(args.outpath, 'train.txt'), sep='\t', header=None, index=False)
    test.to_csv(os.path.join(args.outpath, 'test.txt'), sep='\t', header=None, index=False)