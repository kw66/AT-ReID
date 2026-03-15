# AT-ReID-fast

`AT-ReID-fast` is the deeply accelerated and simplified version of the original `AT-ReID` folder.

- `AT-ReID` stays unchanged.
- `AT-ReID-fast` is the upload-friendly version with deeper training and testing acceleration.
- The default training preset is now `fast-compile`.

## Speedup Summary

Measured on real full AT-USTC runs:

| Item | Original-style baseline | AT-ReID-fast | Speedup |
| --- | ---: | ---: | ---: |
| Full Uni-AT training, 120 epochs | 2h20m16s | 1h25m57s | 1.63x |
| Full AT-USTC test | 170.188 s | 148.531 s | 1.15x |

## Install

```bash
conda env create -f environment.yml
conda activate atreid-fast
```

or

```bash
pip install -r requirements.txt
```

## Prepare Data

Edit:

```text
configs/dataset_roots.example.json
```

or pass the main AT-USTC root directly:

```bash
python train.py --data-root /path/to/atustc ...
```

The default ViT pretrained weight path is:

```text
~/.cache/torch/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth
```

You can override it with:

```bash
python train.py --pretrained-path /path/to/jx_vit_base_p16_224-80ecf9dd.pth ...
```

## Train

Default full Uni-AT training (`fast-compile`):

```bash
python train.py -gpu 0 -v 1 -said -moae -hdw
```

Baseline training:

```bash
python train.py -gpu 0 -v 2
```

Unified embedding training:

```bash
python train.py -gpu 0 -v 3 -ncls 1
```

## Test

AT-USTC evaluation:

```bash
python test.py -gpu 0 -v 1 -said -moae -hdw --checkpoint save_model/atustc_v1/epoch_best.t
```

AT-USTC plus public datasets:

```bash
python test.py -gpu 0 -v 1 -said -moae -hdw --checkpoint save_model/atustc_v1/epoch_best.t -test_all
```

## Notes

- Cross-dataset testing supports AT-USTC, Market1501, CUHK03, MSMT17, SYSU-MM01, RegDB, LLCM, PRCC, LTCC, and DeepChange.
- Project assets are kept under `assets/`.

## Citation

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
  url       = {https://doi.org/10.24963/ijcai.2025/164}
}
```
