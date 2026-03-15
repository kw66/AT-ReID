import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_RELATIVE_ROOTS = {
    "atustc": "data/atustc",
    "market": "data/market1501",
    "cuhk": "data/cuhk03-np/detected",
    "msmt": "data/MSMT17_V1",
    "sysu": "data/sysu",
    "regdb": "data/regdb",
    "llcm": "data/llcm",
    "prcc": "data/prcc",
    "ltcc": "data/ltcc",
    "vc": "data/vc",
    "deepchange": "data/deepchange",
}


DATASET_ENV_KEYS = {
    name: f"ATREID_{name.upper()}_ROOT" for name in DEFAULT_RELATIVE_ROOTS
}


def _load_json_if_exists(config_path):
    if not config_path:
        return {}, None
    path = Path(config_path)
    if not path.is_file():
        return {}, None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f), path.parent


def resolve_dataset_root(name, data_dir=None, data_root_config=None):
    name = name.lower()
    if data_dir:
        return str(Path(data_dir).expanduser())

    env_key = DATASET_ENV_KEYS.get(name)
    if env_key and os.environ.get(env_key):
        return os.environ[env_key]

    config, _ = _load_json_if_exists(data_root_config)
    if name in config:
        resolved = Path(config[name]).expanduser()
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / resolved
        return str(resolved)

    shared_root = os.environ.get("ATREID_DATA_ROOT")
    if shared_root:
        relative = DEFAULT_RELATIVE_ROOTS[name]
        return str(Path(shared_root).expanduser() / relative)

    return str(PROJECT_ROOT / DEFAULT_RELATIVE_ROOTS[name])
