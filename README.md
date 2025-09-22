# AT-ReID

知乎介绍

[\[Task Introduction\]](https://zhuanlan.zhihu.com/p/1944895842541605129)

[\[Dataset Introduction\]](https://zhuanlan.zhihu.com/p/1946682409371304382)

[\[Method Introduction\]](https://zhuanlan.zhihu.com/p/1947080865181078424)

[\[Paper\]]() Towards Anytime Retrieval: A Benchmark for Anytime Person Re-Identification (IJCAI2025 oral)

A Benchmark for Anytime Person Re-Identification (AT-ReID), which aims to retrieve a person at any time, including both daytime and nighttime, ranging from short-term to long-term.

Based on the timestamps of query and gallery images, AT-ReID can be categorized into six scenarios: daytime short-term (DT-ST), daytime long-term (DT-LT), nighttime short-term (NT-ST), nighttime long-term (NT-LT), all-day short-term (AD-ST) and all-day long-term (AD-LT), providing a broader range of scenarios compared to traditional ReID (Tr-ReID), visible-infrared cross-modal ReID (CM-ReID), and long-term cloth-changing ReID (CC-ReID).

<p align="center">   
    <img src="https://github.com/kw66/AT-ReID/blob/main/fig1.png" style="width:90%; display: block; margin-left: auto; margin-right: auto;"> 
</p>

### AT-USTC Dataset

The AT-USTC dataset is constructed to provide conditions for investigations in AT-ReID. Compared to existing datasets, AT-USTC stands out for its long data collection period and the inclusion of both RGB and IR camera footages. Our data collection spans 21 months, and 270 volunteers were photographed on average 29.1 times across different (13) dates or (16) scenes, 4-15 times more than current datasets, leading to the richest intra-identity diversity in scene, clothing, and modality. Importantly, our data collection has obtained the consent of each volunteer. 

Please send a signed [Dataset Release Agreement](https://github.com/kw66/AT-ReID/blob/main/AT-USTC%20Dataset%20Release%20Agreement.pdf) copy to lxlkw@mail.ustc.edu.cn. If your application is approved, we will send the download link for the dataset.

<p align="center">
  <img src="https://github.com/kw66/AT-ReID/blob/main/fig2.png" style="width:90%; display: block; margin-left: auto; margin-right: auto;">
</p>

The AT-USTC dataset is built as the following folder structure:
```
│AT-USTC/
├──p001-d01-c01/
│  ├── cam01-f0-0050.jpg
│  ├── cam01-f0-0100.jpg
│  ├── ......
├──p001-d02-c02/
│  ├── ......
│ .....
```
The image AT-USTC/p001-d01-c01/cam01-f0-0050.jpg denotes the following meaning: 

"p001": person ID (1-270); "d01": capture date ID (1-13); "c01": clothes ID for its owner (1-14); "cam01": camera ID (1-8 for RGB cameras and 9-16 for infrared cameras); "0050": frame ID of the video segment. "f0": image division flag (0 training; 1 validation; 2-10 test) (2,3,4,5 for query; 6,7,8,9 for gallery; 2,6 for DT-ST; 3,7 for DT-LT; 4,8 for NT-ST; 5,9 for NT-LT; 2,4,6,8 for AD-ST; 3,5,7,9 for AD-LT).

We divided the training and testing sets according to ID 1:1. The training set contains 286,087 images from 135 IDs, with 55,060 images (20%) set aside as a validation set. The testing set includes 117,512 images from another 135 IDs. The existing datasets mainly evaluate single scenes, while we constructed separate galleries and query sets for all six scenes covered by AT-ReID for a fine-grained assessment of the model. 

Since the number of images per identity in our dataset is significantly high, the multi-shot evaluation may lead to excessively high rank-1 metrics, while the single-shot evaluation diminishes the relevance of the mAP metric. Therefore, we selected three query images and three gallery images for each identity's video clips (same ID, same camera, same clothing). In this setup, the gallery averages about 25 images per identity, which is comparable to the multi-shot conditions of other datasets (Market1501[1] has 21, MSMT17[2] has 27, and PRCC[5] has 24). We did not perform 10 trials and take the average as done with the SYSU-MM01[3] and LLCM[4] datasets, because the average number of images per identity in these two datasets is 3.1 and 1.6, respectively, which is closer to a single-shot scenario.

### Citation If you use the dataset, please cite the following paper: 
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

[2] Longhui Wei, Shiliang Zhang, Wen Gao, and Qi Tian. Person transfer gan to bridge domain gap for person re-identification. CVPR, 2018.

[3] Ancong Wu, Wei-Shi Zheng, Hong-Xing Yu, Shaogang Gong, and Jianhuang Lai. Rgb-infrared cross-modality person re-identification. ICCV, 2017.

[4]  Yukang Zhang and Hanzi Wang. Diverse embedding expansion network and low-light cross-modality benchmark for visible-infrared person re-identification. CVPR, 2023.

[5] Qize Yang, Ancong Wu, and Wei-Shi Zheng. Person re-identification by contour sketch under moderate clothing change. IEEE TPAMI, 2019.

### Contact 
If you have any questions, please feel free to contact us. E-mail: lxlkw@mail.ustc.edu.cn
