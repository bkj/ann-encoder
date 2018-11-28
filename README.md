### ann-encoder

#### Notes

- This is a link prediction task, and so we measure performance by looking the precision of the top-k predicted links _that are not in the training set_.  This means we have to do some funny postprocessing (see `model.ApproxLinear._compute_dense` and `helpers.__filter_and_rank`).  In other tasks (language modeling, link prediction in a bipartite graphi)

#### Todo / Questions

- [ ] Can we avoid doing some of the synchronizes? 
- [ ] IP search may be slower than L2? Is that true?