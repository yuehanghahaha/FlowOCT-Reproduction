"""Adapt the existing fixed OCTA500_common_v1 manifests to the author loader."""

import csv
import hashlib
import random
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchio as tio
from torch.utils.data import Dataset


COUNTS = {"3M": {"train": 140, "validation": 10, "test": 50},
          "6M": {"train": 180, "validation": 20, "test": 100}}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path, fov, check_counts=True):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty manifest")
    cases, files, patients, groups = set(), set(), {}, {}
    for row in rows:
        if any(not row.get(k, "").strip() for k in ("case_id", "patient_id", "path", "split")):
            raise ValueError("Manifest requires case_id, patient_id, path and split")
        cid, pid, split = row["case_id"], row["patient_id"], row["split"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", cid) or split not in COUNTS[fov]:
            raise ValueError(f"Invalid case or split: {cid}, {split}")
        if row.get("fov", fov) != fov:
            raise ValueError(f"Mixed FOV manifest: {cid}")
        source = (path.parent / row["path"]).resolve()
        if cid in cases or source in files:
            raise ValueError(f"Duplicate case/file: {cid}")
        if not source.is_file() or source.suffix.lower() != ".npz":
            raise FileNotFoundError(source)
        if pid in patients and patients[pid] != split:
            raise ValueError(f"Patient crosses splits: {pid}")
        group = row.get("leakage_group", "").strip()
        if group and group in groups and groups[group] != split:
            raise ValueError(f"Duplicate image group crosses splits: {group}")
        cases.add(cid)
        files.add(source)
        patients[pid] = split
        if group:
            groups[group] = split
        row["resolved_path"] = str(source)
    counts = dict(Counter(row["split"] for row in rows))
    if check_counts and counts != COUNTS[fov]:
        raise ValueError(f"{fov}: expected {COUNTS[fov]}, found {counts}")
    return rows


def inspect_header(row, axis_order, fov):
    """Read NPZ array headers without inflating every large volume into RAM."""
    shapes = []
    with zipfile.ZipFile(row["resolved_path"]) as archive:
        for key in ("oct", "octa"):
            with archive.open(key + ".npy") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f"Unsupported NPY header version {version}")
            if len(shape) != 3 or not np.issubdtype(dtype, np.number):
                raise ValueError(f"Invalid {key} header for {row['case_id']}")
            shapes.append(shape)
    if shapes[0] != shapes[1]:
        raise ValueError(f"Unaligned shapes: {row['case_id']}")
    expected = (256, 304, 304) if fov == "3M" else (256, 400, 400)
    zyx_shape = tuple(shapes[0][axis_order.index(axis)] for axis in "ZYX")
    if zyx_shape != expected:
        raise ValueError(f"Expected common z256 data {expected}, found {zyx_shape}")
    with np.load(row["resolved_path"], allow_pickle=False) as data:
        for key, value in (("axis_order", axis_order), ("case_id", row["case_id"]),
                           ("patient_id", row["patient_id"]), ("intensity_scale", 255)):
            if key in data and data[key].item() != value:
                raise ValueError(f"NPZ {key} conflicts with manifest/config: {row['case_id']}")
    return {"case_id": row["case_id"], "split": row["split"],
            "shape": list(shapes[0]), "zyx_shape": list(zyx_shape)}


class PairedVolumeDataset(Dataset):
    """Original resize/crop/flip/minmax, after an explicit ZYX -> YXZ adapter.

    Y is the B-scan index, X the in-scan lateral coordinate, Z axial depth.
    YXZ follows the upstream loader's [400,400,640] comment; the upstream
    packing script was not released, so this is a documented assumption.
    """

    def __init__(self, rows, split, cfg, limit=0):
        self.rows = [row for row in rows if row["split"] == split]
        if limit:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"No {split} cases")
        self.split, self.cfg = split, cfg
        self.resize = tio.Resize((cfg["load_size"], cfg["load_size"], cfg["crop_size"]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        arrays = []
        with np.load(row["resolved_path"], allow_pickle=False) as data:
            if "axis_order" in data and data["axis_order"].item() != self.cfg["axis_order"]:
                raise ValueError(f"Axis metadata mismatch: {row['case_id']}")
            for key in ("oct", "octa"):
                array = np.asarray(data[key], dtype=np.float32)
                if array.ndim != 3 or not np.isfinite(array).all():
                    raise ValueError(f"Non-finite or non-3D {key}: {row['case_id']}")
                if array.min() < 0 or array.max() > 255:
                    raise ValueError(f"Expected raw [0,255] units for {key}: {row['case_id']}")
                arrays.append(array)
        if arrays[0].shape != arrays[1].shape:
            raise ValueError(f"OCT/OCTA shapes differ: {row['case_id']}")
        raw_shape = arrays[0].shape
        permutation = [self.cfg["axis_order"].index(axis) for axis in "YXZ"]
        volumes = [np.asarray(self.resize(np.transpose(a, permutation).copy()[None]),
                              dtype=np.float32) for a in arrays]
        crop = self.cfg["crop_size"]
        if self.split == "train":
            starts = [random.randint(0, max(0, n - crop)) for n in volumes[0].shape[1:]]
            flip = random.random() > 0.5
        else:
            starts = [(n - crop) // 2 for n in volumes[0].shape[1:]]
            flip = False
        region = (slice(None),) + tuple(slice(s, s + crop) for s in starts)
        tensors, ranges = [], []
        for volume in volumes:
            volume = volume[region]
            if flip:
                volume = np.flip(volume, axis=-1)
            low, high = float(volume.min()), float(volume.max())
            if high <= low:
                raise ValueError(f"Constant crop would produce NaN: {row['case_id']}")
            tensors.append(torch.from_numpy(np.ascontiguousarray((volume - low) / (high - low))))
            ranges.append([low, high])
        return {"A": tensors[0], "B": tensors[1], "case_id": row["case_id"],
                "patient_id": row["patient_id"], "disease": row.get("disease", "unknown"),
                "geometry": {"stored_axis_order": self.cfg["axis_order"],
                             "source_shape": list(raw_shape), "model_axis_order": "YXZ",
                             "resized_shape": [self.cfg["load_size"], self.cfg["load_size"], crop],
                             "crop_start_yxz": starts, "crop_size": crop, "flipped_z": flip,
                             "oct_minmax": ranges[0], "octa_minmax": ranges[1]}}


def collate_volumes(samples):
    return {"A": torch.stack([s["A"] for s in samples]),
            "B": torch.stack([s["B"] for s in samples]), "metadata": samples}
