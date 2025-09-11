# AT-ReID

知乎介绍   [\[Task Introduction\]](https://zhuanlan.zhihu.com/p/1944895842541605129)   [\[Dataset Introduction\]](https://zhuanlan.zhihu.com/p/1946682409371304382)   [\[Method Introduction\]](https://zhuanlan.zhihu.com/p/1947080865181078424)

[\[Paper\]]() Towards Anytime Retrieval: A Benchmark for Anytime Person Re-Identification (IJCAI2025 oral)

### TODO
- [yes] Code of our proposed Uni-AT
- [wait] Code of popular ReID methods for AT-ReID
- [wait] Code for other ReID datasets

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

"p001": person ID (1-270); "d01": capture date ID (1-13); "c01": clothes ID for its owner (1-14); "cam01": camera ID (1-8 for RGB cameras and 9-16 for infrared cameras); "0050": frame ID of the video segment;

"f0": image division flag (0 training; 1 validation; 2-10 test) ([2,3,4,5] query; [6,7,8,9] gallery; [2,6] for DT-ST; [3,7] for DT-LT; [4,8] for NT-ST; [5,9] for NT-LT; [2,4,6,8] for AD-ST; [3,5,7,9] for AD-LT).

### Citation If you use the dataset, please cite the following paper: 
```
@inproceedings{li2025ATreid,
    title={Towards Anytime Retrieval: A Benchmark for Anytime Person Re-Identification.},
    author={Xulin Li, Yan Lu, Bin Liu, Jiaze Li, Qinhong Yang, Tao Gong, Qi Chu, Mang Ye and Nenghai Yu},
    booktitle={International Joint Conferences on Artificial Intelligence (IJCAI)},
    pages={},
    year={2025}
}
```

### Contact 
If you have any questions, please feel free to contact us. E-mail: lxlkw@mail.ustc.edu.cn
