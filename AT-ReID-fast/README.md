# AT-ReID-fast

`AT-ReID-fast` is the deeply accelerated version of the original `AT-ReID` folder.

- `AT-ReID` stays unchanged.
- `AT-ReID-fast` is the optimized upload-friendly version.
- The default runtime already uses the recommended fastest stable path (`fast-compile`).

## Speedup Summary

Benchmarked on real 120-epoch AT-USTC runs:

| Item | AT-ReID project | AT-ReID-fast | Speedup |
| --- | ---: | ---: | ---: |
| Baseline training (120 epochs) | 1h37m49s | 47m45s | 2.05x |
| Full training (120 epochs) | 2h05m21s | 1h31m39s | 1.37x |
| Full AT-USTC test | 174.836 s | 27.792 s | 6.29x |
| Full `test_all` | 1h23m30s | 10m06s | 8.27x |

`test_all` above reports the wall time of the full cross-dataset evaluation stage. Per-dataset timings are omitted here to keep the README short.

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

or pass the AT-USTC root directly:

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

```bash
python train.py -gpu 0 -v 1 -said -moae -hdw
```

## Test

```bash
python test.py -gpu 0 -v 1 -said -moae -hdw --checkpoint save_model/atustc_v1/epoch_best.t
```

```bash
python test.py -gpu 0 -v 1 -said -moae -hdw --checkpoint save_model/atustc_v1/epoch_best.t -test_all
```

## Notes

- Cross-dataset `test_all` supports AT-USTC, Market1501, CUHK03, MSMT17, SYSU-MM01, RegDB, LLCM, PRCC, LTCC, and DeepChange.
- The original `AT-ReID` folder remains the unmodified reference implementation.

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
