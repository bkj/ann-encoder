#!/bin/bash

# run.sh

# --
# Synthetic experiments

python synth.py

CUDA_VISIBLE_DEVICES=7 python synth.py

# --
# ML20 experiments

CUDA_VISIBLE_DEVICES=7 python train.py --cache-path cache/ml20 --emb-dim 800 --batch-size 256
CUDA_VISIBLE_DEVICES=7 python inference.py --cache-path cache/ml20 --emb-dim 800 --batch-size 512
CUDA_VISIBLE_DEVICES=7 python inference.py --cache-path cache/ml20 --emb-dim 800 --benchmark


# --
# Netflix experiments

CUDA_VISIBLE_DEVICES=0 python train.py --cache-path cache/netflix --emb-dim 256 --batch-size 16
# {"epoch": 0, "p_at_01": 0.5907248986942819, "p_at_05": 0.3323615488518685, "p_at_10": 0.24326879783881136, "elapsed": 155.2834813594818}
CUDA_VISIBLE_DEVICES=0 python inference.py --cache-path cache/netflix --emb-dim 256 --batch-size 1024
# {"exact_p_at_01": 0.5840274651058082, "exact_p_at_05": 0.3304029716343989, "exact_p_at_10": 0.24235704637550654}
CUDA_VISIBLE_DEVICES=0 python inference.py --cache-path cache/netflix --emb-dim 256 --batch-size 1024 --benchmark 
# {"exact_time": 1.2870566844940186, "approx_time": 0.7744965553283691, "approx_speedup": 1.6617978164516631}

# Synthetic analog
CUDA_VISIBLE_DEVICES=0 python synth.py --batch-size 1024 --emb-dim 256 \
    --out-dim 480190 --n-toks 480190
# {'exact_time': 3.5207951068878174, 'approx_time': 0.6484885215759277, 'approx_speedup': 5.429232730799519}

CUDA_VISIBLE_DEVICES=0 python synth.py --batch-size 1024 --emb-dim 256 \
    --out-dim 480190 --n-toks 480190 --seq-len 1000
# {'exact_time': 3.875016212463379, 'approx_time': 0.9764342308044434, 'approx_speedup': 3.9685378597091123}






# !! Using larger batch sizes in `inference.py` changes the precision numbers. 
#    Not sure why. BatchNorm?  EmbeddingBag?