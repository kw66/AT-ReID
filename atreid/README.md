Official PyTorch implementation of the paper Towards Anytime Retrieval: A Benchmark for Anytime Person Re-Identification.

### 1. Training.
Train a model by:
```
python train.py -gpu 1 -v 1 -said -moae -hdw
```
-gpu: which gpu to run.

-v: version of training model.

-said: use said loss.

-moae: use moae.

-hdw: use hdw.

You may need mannully define the data path first.

### 2. Testing.
Test a model by
```
python train.py -gpu 1 -v 1 -said -moae -hdw -test
```

### 3. Results.
#### We have made some updates to the results in our paper on the AT-USTC dataset. Please cite the results in the table below.



The results may have some fluctuation due to random spliting and it might be better by finetuning the hyper-parameters.


### 4. Citation
Please kindly cite this paper in your publications if it helps your research:

```
```





