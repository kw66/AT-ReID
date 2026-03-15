import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


if hasattr(Image, "Resampling"):
    _BILINEAR = Image.Resampling.BILINEAR
else:
    _BILINEAR = Image.BILINEAR


@dataclass(frozen=True)
class DecodeCacheInfo:
    mode: str = "off"
    count: int = 0
    bytes: int = 0
    pre_resized: bool = False

    def summary(self) -> str:
        if self.mode == "off":
            return "off"
        gib = self.bytes / (1024 ** 3)
        resize_text = "pre-resized" if self.pre_resized else "decoded-only"
        return f"{self.mode} ({self.count} images, {gib:.2f} GiB, {resize_text})"


class ImageDataset(Dataset):
    def __init__(
        self,
        dataset,
        transform,
        *,
        decode_cache="off",
        cache_resize=None,
        cache_dir=None,
        verbose=False,
    ):
        self.dataset = list(dataset)
        self.transform = transform
        self.decode_cache = str(decode_cache).strip().lower()
        self.cache_resize = tuple(int(item) for item in cache_resize) if cache_resize is not None else None
        self.cache_dir = Path(cache_dir).resolve() if cache_dir is not None else self._infer_cache_dir()
        self.verbose = bool(verbose)
        self._ram_cache = None
        self._disk_cache_paths = None
        self.cache_info = DecodeCacheInfo(mode="off", count=0, bytes=0, pre_resized=self.cache_resize is not None)
        if self.decode_cache not in {"off", "ram", "disk"}:
            raise ValueError(f"Unsupported decode cache mode: {decode_cache}")
        if self.decode_cache == "ram":
            self._build_ram_cache()
        elif self.decode_cache == "disk":
            self._build_disk_cache()

    def __len__(self):
        return len(self.dataset)

    def _infer_cache_dir(self):
        if not self.dataset:
            return None
        sample_paths = [str(Path(img_path).resolve()) for img_path, *_ in self.dataset]
        common = Path(os.path.commonpath(sample_paths))
        if common.is_file():
            common = common.parent
        return common / ".atreid_fast_cache"

    def _load_image(self, img_path):
        with Image.open(Path(img_path)) as handle:
            img = handle.convert("RGB")
        if self.cache_resize is not None:
            height, width = self.cache_resize
            img = img.resize((int(width), int(height)), resample=_BILINEAR)
        return img

    def _disk_cache_path_for(self, img_path):
        if self.cache_dir is None:
            raise RuntimeError("decode_cache='disk' requires a resolvable cache_dir.")
        source = Path(img_path).resolve()
        stat = source.stat()
        resize_token = f"{self.cache_resize[0]}x{self.cache_resize[1]}" if self.cache_resize is not None else "raw"
        digest = hashlib.sha1(
            f"{source}|{stat.st_size}|{stat.st_mtime_ns}|{resize_token}".encode("utf-8")
        ).hexdigest()
        namespace = "pre_resized" if self.cache_resize is not None else "decoded"
        return self.cache_dir / "decode_cache" / namespace / digest[:2] / f"{digest}.npy"

    def _build_ram_cache(self):
        if self.verbose:
            resize_text = (
                f" with pre-resize {self.cache_resize[0]}x{self.cache_resize[1]}"
                if self.cache_resize is not None
                else ""
            )
            print(f"Building RAM decode cache for {len(self.dataset)} images{resize_text}...")
        arrays = []
        total_bytes = 0
        for img_path, *_ in self.dataset:
            array = np.asarray(self._load_image(img_path), dtype=np.uint8).copy()
            arrays.append(array)
            total_bytes += int(array.nbytes)
        self._ram_cache = tuple(arrays)
        self.cache_info = DecodeCacheInfo(
            mode="ram",
            count=len(arrays),
            bytes=total_bytes,
            pre_resized=self.cache_resize is not None,
        )
        if self.verbose:
            print(f"RAM decode cache ready: {self.cache_info.summary()}")

    def _write_disk_cache_file(self, cache_path, array):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(f".tmp-{os.getpid()}.npy")
        np.save(tmp_path, array, allow_pickle=False)
        Path(tmp_path).replace(cache_path)

    def _build_disk_cache(self):
        if self.cache_dir is None:
            raise RuntimeError("decode_cache='disk' requires a cache directory.")
        if self.verbose:
            resize_text = (
                f" with pre-resize {self.cache_resize[0]}x{self.cache_resize[1]}"
                if self.cache_resize is not None
                else ""
            )
            print(f"Building disk decode cache for {len(self.dataset)} images{resize_text} under {self.cache_dir}...")
        cache_paths = []
        total_bytes = 0
        built_count = 0
        for img_path, *_ in self.dataset:
            cache_path = self._disk_cache_path_for(img_path)
            cache_paths.append(cache_path)
            if not cache_path.exists():
                array = np.asarray(self._load_image(img_path), dtype=np.uint8).copy()
                self._write_disk_cache_file(cache_path, array)
                built_count += 1
            total_bytes += int(cache_path.stat().st_size)
        self._disk_cache_paths = tuple(cache_paths)
        self.cache_info = DecodeCacheInfo(
            mode="disk",
            count=len(cache_paths),
            bytes=total_bytes,
            pre_resized=self.cache_resize is not None,
        )
        if self.verbose:
            print(f"Disk decode cache ready: {self.cache_info.summary()} (built {built_count} new files)")

    def __getitem__(self, index):
        img_path, pid, cid, mid, camid = self.dataset[index]
        if self._ram_cache is not None:
            img = Image.fromarray(self._ram_cache[index], mode="RGB")
        elif self._disk_cache_paths is not None:
            array = np.load(self._disk_cache_paths[index], allow_pickle=False)
            img = Image.fromarray(array, mode="RGB")
        else:
            img = self._load_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, cid, mid, camid, index


if __name__ == '__main__':
    pass
