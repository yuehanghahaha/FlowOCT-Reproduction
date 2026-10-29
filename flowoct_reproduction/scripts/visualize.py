"""Display model-space predictions as physical ZYX slices and axial MIP."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.input, allow_pickle=False) as data:
        if data["axis_order"].item() != "ZYX":
            raise ValueError("Expected axis_order=ZYX")
        oct_volume, target, prediction = [data[key] for key in ("oct", "octa_gt", "octa")]
    volumes = [oct_volume, target, prediction, np.abs(prediction - target)]
    names = ["OCT", "Real OCTA", "Generated OCTA", "Absolute error"]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
    for column, (volume, name) in enumerate(zip(volumes, names)):
        # Select anatomical planes using saved axis metadata, not numeric lengths.
        planes = [volume[:, volume.shape[1] // 2, :],
                  volume[:, :, volume.shape[2] // 2], volume.max(axis=0)]
        if column == 3:
            planes[2] = np.abs(prediction.max(axis=0) - target.max(axis=0))
        for row, plane in enumerate(planes):
            axes[row, column].imshow(plane, cmap="magma" if column == 3 else "gray", vmin=0, vmax=1)
            label = ["B-scan (ZX)", "Orthogonal slice (ZY)", "Axial max projection (YX)"][row]
            axes[row, column].set_title(f"{name}\n{label}", fontsize=10)
            axes[row, column].axis("off")
    info_path = args.input.parent / "prediction.json"
    diagnostic = info_path.is_file() and json.loads(info_path.read_text(encoding="utf-8")).get("limited_diagnostic")
    prefix = "DIAGNOSTIC | " if diagnostic else ""
    fig.suptitle(prefix + args.input.stem + " | resized center crop; not original full FOV", fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
