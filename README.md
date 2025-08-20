# AT-ReID
A Benchmark for Anytime Person Re-Identification (AT-ReID), which aims to retrieve a person at any time, including both daytime and nighttime, ranging from short-term to long-term. 

Based on the timestamps of query and gallery images, AT-ReID can be categorized into six scenarios: daytime short-term (DTST), daytime long-term (DT-LT), nighttime short-term (NT-ST), nighttime long-term (NT-LT), all-day short-term (AD-ST) and all-day long-term (AD-LT), providing a broader range of scenarios compared to traditional ReID (Tr-ReID), visible-infrared cross-modal ReID (CM-ReID), and long-term cloth-changing ReID (CC-ReID).

[\[Paper\]]() Towards Anytime Retrieval: A Benchmark for Anytime Person Re-Identification (IJCAI2025)

![image](https://github.com/kw66/AT-ReID/blob/main/fig1.pdf)

### AT-USTC Dataset

The AT-USTC dataset is constructed to provide conditions for investigations in AT-ReID. Compared to existing datasets, AT-USTC stands out for its long data collection period and the inclusion of both RGB and IR camera footages. Our data collection spans over an entire year, and 270 volunteers were photographed on average 29.1 times across different dates or scenes, 4-15 times more than current datasets, leading to the richest intra-identity diversity in scene, clothing, and modality. Importantly, our data collection has obtained the consent of each volunteer. 

Please send a signed [Dataset Release Agreement](https://github.com/kw66/AT-ReID/blob/main/AT-USTC%20Dataset%20Release%20Agreement.pdf) copy to lxlkw@mail.ustc.edu.cn. If your application is approved, we will send the download link for the dataset.

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
