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

Test a model on AT-USTC, Market1501[1], CUHK03[2], MSMT17[3], SYSU-MM01[4], RegDB[5], LLCM[6], PRCC[7], LTCC[8], and DeepChange[9] by 
``` 
python train.py -gpu 0 -v 1 -said -moae -hdw -test -test_all
```

### 3. Results.
#### Our version of IJCAI selected 18 images for each video clip, ultimately choosing 135K images. In the final version, we provide the original 403K images with more perspectives and poses to enhance intra-class diversity. The results have been updated in the arXiv version and the table below.

<p align="center">        
  <img src="https://github.com/kw66/AT-ReID/blob/main/AT-ReID/fig1.png" style="width:70%; display: block; margin-left: auto; margin-right: auto;">  
</p>

### 4. Citation
Please kindly cite this paper in your publications if it helps your research:
```
@inproceedings{li2025ATreid,
  title     = {Towards Anytime Retrieval: A Benchmark for Anytime Person Re-Identification},
  author    = {Li, Xulin and Lu, Yan and Liu, Bin and Li, Jiaze and Yang, Qinhong and Gong, Tao and Chu, Qi and Ye, Mang and Yu, Nenghai},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, {IJCAI-25}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {James Kwok},
  pages     = {1467--1475},
  year      = {2025},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2025/164},
  url       = {https://doi.org/10.24963/ijcai.2025/164},
}
```

###  References.

[1] Liang Zheng, Liyue Shen, Lu Tian, Shengjin Wang, Jingdong Wang, and Qi Tian. Scalable person re-identification: A benchmark. ICCV, 2015.

[2] Wei Li, Rui Zhao, Tong Xiao, and Xiaogang Wang. Deepreid: Deep filter pairing neural network for person re-identification. CVPR, 2014.

[3] Longhui Wei, Shiliang Zhang, Wen Gao, and Qi Tian. Person transfer gan to bridge domain gap for person re-identification. CVPR, 2018.

[4] Ancong Wu, Wei-Shi Zheng, Hong-Xing Yu, Shaogang Gong, and Jianhuang Lai. Rgb-infrared cross-modality person re-identification. ICCV, 2017.

[5] Dat Tien Nguyen, Hyung Gil Hong, Ki Wan Kim, and Kang Ryoung Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 2017.

[6]  Yukang Zhang and Hanzi Wang. Diverse embedding expansion network and low-light cross-modality benchmark for visible-infrared person re-identification. CVPR, 2023.

[7] Qize Yang, Ancong Wu, and Wei-Shi Zheng. Person re-identification by contour sketch under moderate clothing change. IEEE TPAMI, 2019.

[8] Xuelin Qian, Wenxuan Wang, Li Zhang, Fangrui Zhu, Yanwei Fu, Tao Xiang, Yu-Gang Jiang, and Xiangyang Xue. Long-term cloth-changing person re-identification. ACCV, 2020.

[9] Peng Xu and Xiatian Zhu. Deepchange: A long-term person re-identification benchmark with clothes change. ICCV, 2023.





