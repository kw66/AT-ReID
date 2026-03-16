# Compare With Original `AT-ReID`

`AT-ReID-fast` is the optimized sibling folder of the original `AT-ReID`. The original folder stays unchanged.

## Benchmark Summary

Benchmarked on real 120-epoch AT-USTC runs:

| Item | AT-ReID project | AT-ReID-fast | Speedup |
| --- | ---: | ---: | ---: |
| Baseline training (120 epochs) | 1h37m49s | 47m45s | 2.05x |
| Full training (120 epochs) | 2h05m21s | 1h31m39s | 1.37x |
| Full AT-USTC test | 174.836 s | 27.792 s | 6.29x |
| Full `test_all` | 1h23m30s | 10m06s | 8.27x |

`test_all` here reports the wall time of the whole cross-dataset evaluation stage.

## Main Differences

- Hard-coded dataset and pretrained-weight paths were removed.
- A dedicated `test.py` entry was added.
- The default runtime is now `fast-compile`.
- Test-time execution is cleaner and faster:
  - `torch.inference_mode`
  - test AMP
  - GPU distance computation
  - exact GPU ranking when beneficial
  - repeated AT-USTC split evaluation reuses cached features inside one call
- Model selection during training follows the 6-case AT-USTC average (`avg6`).

## Path Handling

Compared with the original folder, `AT-ReID-fast` supports:

- `--data-root`
- `configs/dataset_roots.example.json`
- `--pretrained-path`
- `--no-pretrained`

## Default Usage

```bash
python train.py -gpu 0 -v 1 -said -moae -hdw
```

```bash
python test.py -gpu 0 -v 1 -said -moae -hdw --checkpoint save_model/atustc_v1/epoch_best.t
```

```bash
python test.py -gpu 0 -v 1 -said -moae -hdw --checkpoint save_model/atustc_v1/epoch_best.t -test_all
```
