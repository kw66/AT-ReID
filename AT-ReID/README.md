Official PyTorch implementation of the paper Towards Anytime Retrieval: A Benchmark for Anytime Person Re-Identification.

### 1. Training.

You may need mannully define the data path first.

Train a full Uni-AT model by:
```
python train.py -gpu 0 -v 1 -said -moae -hdw
```
-gpu 0: run on gpu 0.

-v 1: version 1 of the training model.

-said: use said loss.

-moae: use moae.

-hdw: use hdw.

Train a baseline model by:
``` 
python train.py -gpu 0 -v 2
```

Train a unified embedding model by: 
```
python train.py -gpu 0 -v 3 -ncls 1 
```
-ncls 1: use only 1 CLS token rather than 6.

### 2. Testing.
Test a model by
```
python train.py -gpu 0 -v 1 -said -moae -hdw -test
```

Test a model on AT-USTC, Market1501, CUHK03, MSMT17, SYSU-MM01, RegDB, LLCM, PRCC, LTCC, and DeepChange by 
``` 
python train.py -gpu 0 -v 1 -said -moae -hdw -test -test_all
```

### 3. Results.
#### Our version of IJCAI selected 18 images for each video clip, ultimately choosing 135K images. This is based on existing datasets, which typically generate an average of 7-50 images per video clip. However, we later realized that including images with more perspectives and poses could enhance intra-class diversity, and we should not abandon our strengths. Therefore, we open-sourced the original 403K images, which indeed improve generalization, and the results have been updated in the arXiv version and the table below.





### 4. Citation
Please kindly cite this paper in your publications if it helps your research:
```
@inproceedings{li2025ATreid,
    title={Towards Anytime Retrieval: A Benchmark for Anytime Person Re-Identification.},
    author={Xulin Li, Yan Lu, Bin Liu, Jiaze Li, Qinhong Yang, Tao Gong, Qi Chu, Mang Ye and Nenghai Yu},
    booktitle={International Joint Conferences on Artificial Intelligence (IJCAI)},
    pages={},
    year={2025}
}
```





