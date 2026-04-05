# AT-ReID

Official repository for **Towards Anytime Retrieval: A Benchmark for Anytime Person Re-Identification**.

## 🧭 Navigation

<table>
  <tr>
    <td align="center" width="33%">
      <strong>🎯 任务</strong><br><br>
      <a href="#at-reid-task">
        <img src="https://img.shields.io/badge/GitHub-Overview-24292f?style=flat-square&logo=github&logoColor=white" alt="Task overview">
      </a>
      <a href="https://zhuanlan.zhihu.com/p/1944895842541605129">
        <img src="https://img.shields.io/badge/Zhihu-Intro-0084ff?style=flat-square" alt="Task introduction">
      </a>
    </td>
    <td align="center" width="33%">
      <strong>🗂️ 数据集</strong><br><br>
      <a href="#at-ustc-dataset">
        <img src="https://img.shields.io/badge/GitHub-Overview-24292f?style=flat-square&logo=github&logoColor=white" alt="Dataset overview">
      </a>
      <a href="https://zhuanlan.zhihu.com/p/1946682409371304382">
        <img src="https://img.shields.io/badge/Zhihu-Intro-0084ff?style=flat-square" alt="Dataset introduction">
      </a>
    </td>
    <td align="center" width="33%">
      <strong>🧠 方法</strong><br><br>
      <a href="#method-navigation">
        <img src="https://img.shields.io/badge/GitHub-Navigation-24292f?style=flat-square&logo=github&logoColor=white" alt="Method navigation">
      </a>
      <a href="https://zhuanlan.zhihu.com/p/1947080865181078424">
        <img src="https://img.shields.io/badge/Zhihu-Intro-0084ff?style=flat-square" alt="Method introduction">
      </a>
    </td>
  </tr>
</table>

## 🎯 AT-ReID Task

AT-ReID is a benchmark for **anytime person re-identification**, which aims to retrieve a person at any time, including both daytime and nighttime, ranging from short-term to long-term.

Based on the timestamps of query and gallery images, AT-ReID can be categorized into six scenarios: daytime short-term (DT-ST), daytime long-term (DT-LT), nighttime short-term (NT-ST), nighttime long-term (NT-LT), all-day short-term (AD-ST), and all-day long-term (AD-LT).

<p align="center">
  <img src="./fig1.png" style="width:90%; display: block; margin-left: auto; margin-right: auto;">
</p>

## 🗂️ AT-USTC Dataset

The AT-USTC dataset is constructed to support investigations in AT-ReID. Compared with existing datasets, AT-USTC stands out for its long collection period and the inclusion of both RGB and IR camera footage.

Our data collection spans **21 months**, and **270 volunteers** were photographed on average **29.1 times** across different dates or scenes, leading to rich intra-identity diversity in scene, clothing, and modality.

<p align="center">
  <img src="./fig2.png" style="width:90%; display: block; margin-left: auto; margin-right: auto;">
</p>

### 📥 Dataset Access

Please send a signed [Dataset Release Agreement](./AT-USTC%20Dataset%20Release%20Agreement.pdf) copy to **lxlkw@mail.ustc.edu.cn**. If your application is approved, we will send the download link for the dataset.

### 📊 Split Summary

| Split | Content |
| --- | --- |
| Training | 286,087 images from 135 IDs |
| Validation | 55,060 images |
| Testing | 117,512 images from another 135 IDs |

## 🧠 Method Navigation

The root README only provides the task overview, dataset overview, and entry links.

- [`AT-ReID-fast/`](./AT-ReID-fast): 🚀 recommended entry point for training and evaluation.
- [`AT-ReID/`](./AT-ReID): 🧪 original reference implementation.

## 📚 Citation

If this project helps your research, please cite:

```bibtex
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

## 📮 Contact

If you have any questions, please feel free to contact us: **lxlkw@mail.ustc.edu.cn**
