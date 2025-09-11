Official PyTorch implementation of the paper Towards Anytime Retrieval: A Benchmark for Anytime Person Re-Identification.

### 1. Training.
Train a model by:
```
python train.py -gpu 1 -v 1 -moe 10
```
-gpu: which gpu to run.

-v: version of training model.

-hdw: do not use hdw.

-said: do not use said loss.

-moe 10: our moae.

-moe 1: do not use moae.

You may need mannully define the data path first.




