from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _read_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        img = tifffile.imread(str(path))
    else:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")

    img = np.asarray(img)

    if img.ndim == 3:
        # OpenCV reads BGR; convert to RGB for consistency.
        if suffix not in {".tif", ".tiff"} and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def _to_channel_first_float(img: np.ndarray, channels: int) -> np.ndarray:
    img = img.astype(np.float32)

    if channels == 1:
        if img.ndim == 3:
            img = img.mean(axis=2)
        img = img[None, :, :]
    elif channels == 3:
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        if img.shape[-1] > 3:
            img = img[:, :, :3]
        img = np.transpose(img, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported channel count: {channels}")

    return img


def _normalize_minmax_per_image(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    lo = np.nanmin(img)
    hi = np.nanmax(img)

    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Image contains only NaN/Inf values.")

    if hi - lo < eps:
        return np.zeros_like(img, dtype=np.float32)

    return ((img - lo) / (hi - lo)).astype(np.float32)


def _resize_chw(img: np.ndarray, size: int) -> np.ndarray:
    c, h, w = img.shape

    resized_channels = []
    for i in range(c):
        resized = cv2.resize(
            img[i],
            (size, size),
            interpolation=cv2.INTER_AREA if h > size or w > size else cv2.INTER_LINEAR,
        )
        resized_channels.append(resized)

    return np.stack(resized_channels, axis=0).astype(np.float32)


class PairedImageDataset(Dataset):
    """
    Dataset for paired XCT -> OM image translation.

    Assumption:
        xct_dir/sample_001.tif corresponds to om_dir/sample_001.tif

    The pairing is based on filename stem.
    """

    def __init__(
        self,
        xct_dir: str | Path,
        om_dir: str | Path,
        image_size: int,
        input_channels: int = 1,
        output_channels: int = 1,
        normalize_mode: str = "minmax_per_image",
        augment: bool = False,
    ) -> None:
        self.xct_dir = Path(xct_dir)
        self.om_dir = Path(om_dir)
        self.image_size = int(image_size)
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.normalize_mode = normalize_mode
        self.augment = augment

        if not self.xct_dir.exists():
            raise FileNotFoundError(f"XCT directory does not exist: {self.xct_dir}")
        if not self.om_dir.exists():
            raise FileNotFoundError(f"OM directory does not exist: {self.om_dir}")

        self.pairs = self._collect_pairs()

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No paired images found between {self.xct_dir} and {self.om_dir}. "
                "Make sure filenames match."
            )

    def _collect_pairs(self) -> List[Tuple[Path, Path]]:
        xct_files = {
            p.stem: p
            for p in self.xct_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        }
        om_files = {
            p.stem: p
            for p in self.om_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        }

        common = sorted(set(xct_files.keys()) & set(om_files.keys()))
        missing_om = sorted(set(xct_files.keys()) - set(om_files.keys()))
        missing_xct = sorted(set(om_files.keys()) - set(xct_files.keys()))

        if missing_om:
            print(f"Warning: {len(missing_om)} XCT files have no matching OM file.")
        if missing_xct:
            print(f"Warning: {len(missing_xct)} OM files have no matching XCT file.")

        return [(xct_files[k], om_files[k]) for k in common]

    def __len__(self) -> int:
        return len(self.pairs)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        if self.normalize_mode == "minmax_per_image":
            return _normalize_minmax_per_image(img)
        raise ValueError(f"Unknown normalize_mode: {self.normalize_mode}")

    def _augment_pair(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Only geometry-preserving augmentations are allowed.
        # They must be applied identically to XCT and OM.
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=2).copy()
            y = np.flip(y, axis=2).copy()

        if np.random.rand() < 0.5:
            x = np.flip(x, axis=1).copy()
            y = np.flip(y, axis=1).copy()

        k = np.random.randint(0, 4)
        if k > 0:
            x = np.rot90(x, k=k, axes=(1, 2)).copy()
            y = np.rot90(y, k=k, axes=(1, 2)).copy()

        return x, y

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        xct_path, om_path = self.pairs[idx]

        x = _read_image(xct_path)
        y = _read_image(om_path)

        x = _to_channel_first_float(x, self.input_channels)
        y = _to_channel_first_float(y, self.output_channels)

        x = self._normalize(x)
        y = self._normalize(y)

        x = _resize_chw(x, self.image_size)
        y = _resize_chw(y, self.image_size)

        if self.augment:
            x, y = self._augment_pair(x, y)

        return {
            "xct": torch.from_numpy(x),
            "om": torch.from_numpy(y),
            "xct_path": str(xct_path),
            "om_path": str(om_path),
        }