"""Maintainer utility: build exact reference copies and adaptation patches."""

import difflib
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    upstream = root.parent / "upstream"
    pinned = "2309126956d2b80a95f2d307d85a50799cc658a6"
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip()
    if head != pinned or status:
        raise RuntimeError("Upstream must be the pinned, unmodified author checkout")
    files = subprocess.check_output(["git", "-C", str(upstream), "ls-files"], text=True).splitlines()
    original_hashes = {}
    for relative in files:
        source = upstream / relative
        destination = root / "reference/upstream" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        original_hashes[relative] = digest(source)
    mappings = {"models/FlowOCT_model.py": "flowoct/model.py",
                "util/DWT_IDWT_layer.py": "flowoct/wavelets.py",
                "util/DWT_IDWT_Functions.py": "flowoct/wavelet_functions.py",
                "util/auto_crop.py": "flowoct/roi.py",
                "datasets/OCTA500_dataset.py": "flowoct/data.py",
                "requirements.txt": "requirements.txt"}
    for original, adapted in mappings.items():
        patch = "".join(difflib.unified_diff(
            (upstream / original).read_text(encoding="utf-8").splitlines(keepends=True),
            (root / adapted).read_text(encoding="utf-8").splitlines(keepends=True),
            fromfile="a/upstream/" + original, tofile="b/" + adapted))
        (root / "provenance" / (Path(adapted).stem + ".patch")).write_text(patch, encoding="utf-8")
    current = {str(path.relative_to(root)).replace("\\", "/"): digest(path)
               for base in ("flowoct", "configs", "scripts", "tests")
               for path in sorted((root / base).rglob("*"))
               if path.is_file() and "__pycache__" not in path.parts}
    for name in ("run.py", "a100.py", "requirements.txt"):
        current[name] = digest(root / name)
    report = {"upstream_url": "https://github.com/cnu-medilab/FlowOCT", "upstream_commit": pinned,
              "upstream_clean": True, "upstream_files_sha256": original_hashes,
              "adapted_files_sha256": current,
              "copied_without_changes": ["flowoct/wavelet_functions.py", "flowoct/roi.py"],
              "new_workflow": ["flowoct/runner.py", "flowoct/metrics.py", "run.py", "a100.py", "configs/", "scripts/", "tests/"]}
    (root / "provenance/source_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    dataset = root.parent / "data/OCTA500_common_v1"
    shutil.copyfile(dataset / "protocol.json", root / "provenance/dataset_protocol.json")
    hashes = {name: digest(dataset / name) for name in ("split_3M_z256.csv", "split_6M_z256.csv")}
    (root / "provenance/split_manifest_sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print("Reference, source fingerprints and patches written; upstream unchanged.")


if __name__ == "__main__":
    main()
