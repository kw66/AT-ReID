# AT-ReID
A Benchmark for Anytime Person Re-Identification (AT-ReID), which aims to retrieve a person at any time, including both daytime and nighttime, ranging from short-term to long-term. 

Based on the timestamps of query and gallery images, AT-ReID can be categorized into six scenarios: daytime short-term (DTST), daytime long-term (DT-LT), nighttime short-term (NT-ST), nighttime long-term (NT-LT), all-day short-term (AD-ST) and all-day long-term (AD-LT), providing a broader range of scenarios compared to traditional ReID (Tr-ReID), visible-infrared cross-modal ReID (CM-ReID), and long-term cloth-changing ReID (CC-ReID).

[\[Paper\]]() Towards Anytime Retrieval: A Benchmark for Anytime Person Re-Identification (IJCAI2025)

### AT-USTC Dataset
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
Xulin Li, Yan Lu, Bin Liu, Jiaze Li, Qinhong Yang, Tao Gong, Qi Chu, Mang Ye and Nenghai Yu, “Towards Anytime Retrieval: A Benchmark for Anytime Person Re-Identification.” International Joint Conferences on Artificial Intelligence, 2025. 
```

