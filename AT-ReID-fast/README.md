# AT-ReID-fast

`AT-ReID-fast` is the simplified and optimized sibling of the original `AT-ReID` folder.

- `AT-ReID` stays unchanged.
- `AT-ReID-fast` is the easier-to-run version for upload and reuse.
- The default training preset is now `fast-compile`.

For the concise change list, see [COMPARE_WITH_AT-ReID.md](COMPARE_WITH_AT-ReID.md).

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

Same setting without compile:

```bash
python train.py -gpu 0 -v 1 -said -moae -hdw --runtime-mode fast
```

Conservative fallback:

```bash
python train.py -gpu 0 -v 1 -said -moae -hdw --runtime-mode strict
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

Useful optional test acceleration flags:

```bash
--test-distance-device cuda --test-rank-device auto --test-amp
```

Optional manual experiments:

```bash
--decode-cache disk
--test-batch-auto-tune
```

These two are kept optional and are not enabled by default.

## Runtime Modes

- `fast-compile`
  Default preset for long stable runs. Uses AMP, worker/prefetch speedups, and `torch.compile`.
- `fast`
  Recommended when you want most of the speedup but do not want compile warmup.
- `strict`
  Conservative fallback close to the original behavior.

## Notes

- This folder does not add neighbor-module acceleration.
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
