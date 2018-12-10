#!/bin/bash

# run_test.sh

# ml20 analog
python synth.py --emb-dim 800 --out-dim 25878
# {'exact_time': 1.1707470417022705, 'approx_time': 0.9718101024627686, 'approx_speedup': 1.204707626248538}

python synth.py --emb-dim 800 --out-dim 100000
# {'exact_time': 3.7346084117889404, 'approx_time': 1.580601692199707, 'approx_speedup': 2.3627764225606542}

python synth.py --emb-dim 800 --out-dim 400000
# {'exact_time': 14.330285787582397, 'approx_time': 4.0452258586883545, 'approx_speedup': 3.542518091246684}























# =========================================================================

# L2
python synth.py --emb-dim 64
# {'exact_time': 1.376793384552002, 'approx_time': 0.5546786785125732, 'approx_speedup': 2.4821458582904468}
python synth.py --emb-dim 128
# {'exact_time': 2.356105089187622, 'approx_time': 0.7755272388458252, 'approx_speedup': 3.038068775887336}
python synth.py --emb-dim 256
# {'exact_time': 4.437787771224976, 'approx_time': 1.0572614669799805, 'approx_speedup': 4.197436405113028}
python synth.py --emb-dim 512
# {'exact_time': 8.645541191101074, 'approx_time': 1.2734665870666504, 'approx_speedup': 6.7889815711737915}
CUDA_VISIBLE_DEVICES=7 python synth.py --emb-dim 800
# {'exact_time': 13.332379341125488, 'approx_time': 2.2359402179718018, 'approx_speedup': 5.962761988877839}
python synth.py --emb-dim 1024
# {'exact_time': 17.026272773742676, 'approx_time': 2.2736849784851074, 'approx_speedup': 7.488404477689255}

# IP -- worse
python synth.py --emb-dim 64
# {'exact_time': 1.3462111949920654, 'approx_time': 0.5176374912261963, 'approx_speedup': 2.6006833311148254}
python synth.py --emb-dim 128
# {'exact_time': 2.3564863204956055, 'approx_time': 0.8209221363067627, 'approx_speedup': 2.870535725954687}
python synth.py --emb-dim 256
# {'exact_time': 4.447732925415039, 'approx_time': 1.1509416103363037, 'approx_speedup': 3.8644296856340237}
python synth.py --emb-dim 512
# {'exact_time': 8.639007568359375, 'approx_time': 1.9481420516967773, 'approx_speedup': 4.434485442596468}
python synth.py --emb-dim 800
# {'exact_time': 13.327322006225586, 'approx_time': 4.02344012260437, 'approx_speedup': 3.312419621047776}
python synth.py --emb-dim 1024
# {'exact_time': 17.019128799438477, 'approx_time': 4.465805768966675, 'approx_speedup': 3.8109872394599162}


# L2 is faster than IP, and gets better speedups in high dimensions -- why?
# L2 is effected by the norm of the initialization in `basenet-rec.py` -- why?
#   IP is not


CUDA_VISIBLE_DEVICES=7 python basenet-rec.py --emb-dim 800 --cache-path cache/ml20
# IP, no init
# exact=False 2.798499345779419
# exact=True 9.114121913909912

CUDA_VISIBLE_DEVICES=7 python basenet-rec.py --emb-dim 800
# IP, w/ init
# exact=False 2.78657603263855
# exact=True 9.10705018043518

CUDA_VISIBLE_DEVICES=7 python basenet-rec.py --emb-dim 800
# L2, w/ init
# exact=False 22.021682739257812 # !!
# exact=True 9.221946954727173

# --
# Netflix simulation

python synth.py --out-dim 480189 --n-toks 17770 --emb-dim 128
# {'exact_time': 2.8564934730529785, 'approx_time': 0.8344957828521729, 'approx_speedup': 3.4230172659350555}

python synth.py --out-dim 480189 --n-toks 17770 --emb-dim 512
# {'exact_time': 10.347851276397705, 'approx_time': 1.3666396141052246, 'approx_speedup': 7.571748374331092}

# --
# Netflix 

# CUDA_VISIBLE_DEVICES=4 python basenet-rec.py \
#     --train-path ../../data/netflix/edgelist-train.tsv \
#     --test-path ../../data/netflix/edgelist-test.tsv \
#     --cache-path ./cache/netflix

python train.py --cache-path ./cache/ml20 --epochs 1
python inference.py --cache-path ./cache/ml20 --benchmark

python train.py --cache-path ./cache/netflix --epochs 1 --batch-size 16 --emb-dim 256
CUDA_VISIBLE_DEVICES=7 python inference.py --cache-path ./cache/netflix --benchmark --emb-dim 256
# Speedup is less than in synthetic case, possibly due to batch sizes.  Should deconflict.

