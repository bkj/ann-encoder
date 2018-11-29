### ann-encoder

#### Notes

- This is a recommender system task, and so we measure performance by looking the precision of the top-k predicted links _that are not in the training set_.  This means we have to do some funny postprocessing (see `model.ApproxLinear._compute_dense` and `helpers.__filter_and_rank`).  In other tasks (language modeling), we wouldn't need to do this.  This means that the performance of even a flat faiss index is degraded compared to the full torch matmul -- if we get the top 32 predictions, these may all be in the training set, and so we default to random predictions. Thus, for now, to determine how much ANN hurts results, we should be comparing it to `ApproxLinear` w/ `flat=True`, rather than to the exact matmul.

- Performance from `synth.py` and in real world datasets can vary, because the dimensions of a batch from `RaggedAutoencoderDataset` can vary.  `synth.py` gives an idealized prediction.

#### Todo + Questions

- Can we avoid doing some of the synchronizes? 
- IP search may be slower than L2? Is that true?
- [BUG] Changing the batch size in `inference.py` changes the precision slightly. 
- [TODO] Remove dependency on custom basenet