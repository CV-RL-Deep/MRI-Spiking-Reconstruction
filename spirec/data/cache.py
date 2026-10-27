import os
import random
import threading
import time

from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np

from spirec.configs.config import Config
from spirec.core.utils import logger


class MedicalVolume:
    """Standardized in-RAM container for normalized (Z, H, W) volumetric data."""

    def __init__(
        self,
        t1_data: np.ndarray,
        seg_data: Optional[np.ndarray] = None,
        path: str = "",
    ):
        self.t1 = t1_data  # Shape: (Z, H, W) in [0.0, 1.0]
        self.seg = seg_data  # Shape: (Z, H, W) or None
        self.path = path
        self.shape = t1_data.shape

        self.indices = {
            "any": {
                0: np.array([], dtype=int),
                1: np.array([], dtype=int),
                2: np.array([], dtype=int),
            },
            "clean": {
                0: np.array([], dtype=int),
                1: np.array([], dtype=int),
                2: np.array([], dtype=int),
            },
            "tumor": {
                0: np.array([], dtype=int),
                1: np.array([], dtype=int),
                2: np.array([], dtype=int),
            },
        }
        self._analyze_spatial_occupancy()

    def _analyze_spatial_occupancy(self) -> None:
        """Computes valid slice coordinate indices across Axial, Coronal, and Sagittal views."""
        for axis in (0, 1, 2):
            reduce_axes = tuple(i for i in range(3) if i != axis)
            tissue_thresh = (
                self.shape[reduce_axes[0]] * self.shape[reduce_axes[1]] * 0.01
            )
            tissue_mask = np.sum(self.t1, axis=reduce_axes) > tissue_thresh
            self.indices["any"][axis] = np.where(tissue_mask)[0]

            if self.seg is not None:
                tumor_mask = np.max(self.seg, axis=reduce_axes) > 0
                self.indices["tumor"][axis] = np.where(tumor_mask)[0]
                self.indices["clean"][axis] = np.where(
                    tissue_mask & (~tumor_mask)
                )[0]
            else:
                self.indices["clean"][axis] = self.indices["any"][axis]


class VolumeLoader:
    """Standardized I/O engine for NIfTI volumes: Canonical RAS+, bounding crop, and scaling."""

    @staticmethod
    def load(
        t1_path: str, seg_path: Optional[str] = None
    ) -> Optional[MedicalVolume]:
        try:
            t1_img = nib.load(t1_path)
            t1_img = nib.as_closest_canonical(t1_img)
            t1_data = t1_img.get_fdata(dtype=np.float32)

            # Transpose Canonical (X=Sag, Y=Cor, Z=Ax) -> Logical (Z=Ax, Y=Cor, X=Sag)
            t1_data = np.transpose(t1_data, (2, 1, 0))

            seg_data = None
            if seg_path and os.path.exists(seg_path):
                seg_img = nib.load(seg_path)
                seg_img = nib.as_closest_canonical(seg_img)
                seg_data = seg_img.get_fdata(dtype=np.float32)
                seg_data = np.transpose(seg_data, (2, 1, 0))

            # Bounding box crop around brain matter
            threshold = np.mean(t1_data) * 0.1
            mask = t1_data > threshold
            if np.any(mask):
                coords = np.argwhere(mask)
                z_min, y_min, x_min = coords.min(axis=0)
                z_max, y_max, x_max = coords.max(axis=0) + 1

                t1_data = t1_data[z_min:z_max, y_min:y_max, x_min:x_max]
                if seg_data is not None:
                    seg_data = seg_data[z_min:z_max, y_min:y_max, x_min:x_max]

            # Robust 99th-percentile scaling
            p99 = float(np.percentile(t1_data, 99))
            if p99 > 0:
                t1_data = np.clip(t1_data, 0.0, p99) / p99

            return MedicalVolume(t1_data, seg_data, path=t1_path)
        except Exception as e:
            logger.warning(f"VolumeLoader failed for {t1_path}: {e}")
            return None


class DataLoaderBase(ABC):
    """Abstract thread-safe volume pool container."""

    def __init__(self, config: Config):
        self.cfg = config
        self.lock = threading.Lock()
        self.pool: Dict[
            str, Union[Dict[str, MedicalVolume], List[MedicalVolume]]
        ] = {}

    @abstractmethod
    def get_volume(self, collection_id: str) -> Optional[MedicalVolume]:
        pass

    def get_matched_glioma_stack(
        self,
        axis: int,
        target_norm_depth: float,
        neighborhood: int,
        tolerance: float = 0.05,
    ) -> Optional[np.ndarray]:
        """
        Retrieves an (N+1, H, W) tumor segmentation stack from BraTS pool matching target normalized depth.
        """
        with self.lock:
            pool_ref = self.pool.get("brats", {})
            brats_vols = (
                list(pool_ref.values())
                if isinstance(pool_ref, dict)
                else list(pool_ref)
            )

        if not brats_vols:
            return None

        candidates_pool = [v for v in brats_vols if v.seg is not None]
        if not candidates_pool:
            return None

        random.shuffle(candidates_pool)

        for vol in candidates_pool:
            tumor_indices = vol.indices["tumor"][axis]
            if len(tumor_indices) == 0:
                continue

            axis_length = vol.shape[axis]
            valid_starts = []

            for t_idx in tumor_indices:
                norm_pos = t_idx / max(1.0, float(axis_length - 1))
                if abs(norm_pos - target_norm_depth) <= tolerance:
                    s_idx = t_idx - neighborhood
                    if s_idx >= 0 and (s_idx + neighborhood) < axis_length:
                        valid_starts.append(s_idx)

            if valid_starts:
                start_z = random.choice(valid_starts)
                if axis == 0:
                    view = vol.seg
                elif axis == 1:
                    view = np.moveaxis(vol.seg, 1, 0)
                else:
                    view = np.moveaxis(vol.seg, 2, 0)

                seg_stack = view[start_z : start_z + neighborhood + 1]
                if seg_stack.shape[0] == neighborhood + 1:
                    return seg_stack.copy()

        return None


class StaticLoader(DataLoaderBase):
    """Loads all volumes into persistent RAM upfront (Cluster & Evaluation mode)."""

    def __init__(
        self, config: Config, ixi_paths: List[str], brats_entries: List[Dict]
    ):
        super().__init__(config)
        self.pool = {"ixi": [], "brats": []}

        for path in ixi_paths:
            vol = VolumeLoader.load(path)
            if vol:
                self.pool["ixi"].append(vol)

        for entry in brats_entries:
            vol = VolumeLoader.load(entry["t1"], entry.get("seg"))
            if vol:
                self.pool["brats"].append(vol)

        logger.info(
            f"StaticLoader initialized: {len(self.pool['ixi'])} IXI, {len(self.pool['brats'])} BraTS in RAM"
        )

    def get_volume(self, collection_id: str) -> Optional[MedicalVolume]:
        collection = self.pool.get(collection_id, [])
        if not collection:
            return None
        return random.choice(collection)


class ActiveLoader(DataLoaderBase):
    """Dynamic background worker rolling volumes into RAM under constrained capacity (Kaggle mode)."""

    def __init__(
        self, config: Config, ixi_paths: List[str], brats_entries: List[Dict]
    ):
        super().__init__(config)
        self.files = {}
        if ixi_paths:
            self.files["ixi"] = ixi_paths
        if brats_entries:
            self.files["brats"] = brats_entries

        self.pool = {k: OrderedDict() for k in self.files.keys()}
        self.active = False
        self.worker_thread: Optional[threading.Thread] = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self) -> None:
        if self.active:
            return
        self.active = True
        self.worker_thread = threading.Thread(
            target=self._preload_loop, daemon=True
        )
        self.worker_thread.start()

        # Warmup loop: wait until minimum volumes are cached
        warmup_target = min(5, self.cfg.data.ixi_cache_size // 4)
        while len(self.pool.get("ixi", {})) < warmup_target and self.active:
            time.sleep(0.5)

    def stop(self) -> None:
        self.active = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)

    def get_volume(self, collection_id: str) -> Optional[MedicalVolume]:
        with self.lock:
            col_dict = self.pool.get(collection_id, {})
            if not col_dict:
                return None
            key = random.choice(list(col_dict.keys()))
            return col_dict[key]

    def _preload_loop(self) -> None:
        indices = {k: 0 for k in self.files.keys()}
        for k in self.files:
            random.shuffle(self.files[k])

        while self.active:
            for col in list(self.files.keys()):
                file_list = self.files[col]
                idx = indices[col]
                item = file_list[idx]
                indices[col] = (idx + 1) % len(file_list)

                if col == "ixi":
                    t1_p, seg_p = item, None
                    key = os.path.basename(item)
                    limit = self.cfg.data.ixi_cache_size
                else:
                    t1_p, seg_p = item["t1"], item.get("seg")
                    key = item["id"]
                    limit = self.cfg.data.brats_cache_size

                vol = VolumeLoader.load(t1_p, seg_p)
                if vol:
                    with self.lock:
                        if len(self.pool[col]) >= limit:
                            self.pool[col].popitem(last=False)
                        self.pool[col][key] = vol

            # Adaptive throttle
            total_cached = sum(len(p) for p in self.pool.values())
            target_capacity = (
                self.cfg.data.ixi_cache_size + self.cfg.data.brats_cache_size
            )
            time.sleep(0.05 if total_cached >= target_capacity else 0.005)
