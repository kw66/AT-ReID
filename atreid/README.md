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
#### Our version of IJCAI selected 18 images for each video clip, ultimately choosing 135K images. This is based on existing datasets, which typically generate an average of 7-50 images per video clip. However, we later realized that including images with more perspectives and poses could enhance intra-class diversity, and we should not abandon our strengths. Therefore, we open-sourced the original 403K images, which indeed improve generalization, and the results have been updated in the arXiv version and the table below.





### 4. Citation
Please kindly cite this paper in your publications if it helps your research:

```
```





