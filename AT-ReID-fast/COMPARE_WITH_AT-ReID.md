# Compare With Original `AT-ReID`

`AT-ReID-fast` is a sibling folder of the original `AT-ReID`. The original folder is unchanged. This one keeps the same main method but makes the code easier to run, test, and upload.

## Main Differences

- Hard-coded dataset and pretrained-weight paths were removed.
- A dedicated `test.py` entry was added.
- Runtime presets are explicit:
  - `strict`
  - `fast`
  - `fast-compile`
- The default preset is now `fast-compile`.
- Test-time feature extraction is cleaner and faster:
  - `torch.inference_mode`
  - optional test AMP
  - optional GPU distance and exact ranking
  - repeated AT-USTC splits reuse cached features inside one evaluation call
- Model selection during training follows the 6-case AT-USTC average (`avg6`).

## Path Handling

Compared with the original folder, `AT-ReID-fast` supports:

- `--data-root`
- `configs/dataset_roots.example.json`
- `--pretrained-path`
- `--no-pretrained`

## Recommended Usage

Default full run:

```bash
python train.py -gpu 0 -v 1 -said -moae -hdw
```

Without compile:

```bash
python train.py -gpu 0 -v 1 -said -moae -hdw --runtime-mode fast
```

Conservative fallback:

```bash
python train.py -gpu 0 -v 1 -said -moae -hdw --runtime-mode strict
```

`AT-ReID-fast` keeps the original AT-ReID scope, while making the code cleaner and the runtime path faster.
