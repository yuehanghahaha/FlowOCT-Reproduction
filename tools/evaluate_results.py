"""Transparent patient-file metrics, not an implementation of the paper's entire table."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity


def psnr(mse):
    return float('inf') if mse == 0 else float(-10 * np.log10(mse))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=0.05, help='illustrative foreground threshold; set on validation then freeze')
    args = parser.parse_args()
    files = sorted(args.input.rglob('*.npz'))
    if not files:
        raise ValueError('No result files')
    rows = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            pred, gt = data['octa'].astype('float64'), data['octa_gt'].astype('float64')
        if pred.shape != gt.shape or pred.ndim != 3 or min(gt.shape) < 7:
            raise ValueError(f'Invalid paired result shapes: {path}')
        if not np.isfinite(pred).all() or not np.isfinite(gt).all():
            raise ValueError(f'NaN/Inf: {path}')
        err = pred - gt
        foreground = gt > args.threshold
        background = ~foreground
        rows.append({'file': str(path.relative_to(args.input)), 'mae_full': float(abs(err).mean()),
                     'psnr_full_range1': psnr(float((err ** 2).mean())),
                     'ssim_3d_range1_win7': float(structural_similarity(gt, pred, data_range=1.0, win_size=7)),
                     'mae_fg_gt_threshold': float(abs(err[foreground]).mean()) if foreground.any() else float('nan'),
                     'psnr_fg_range1': psnr(float((err[foreground] ** 2).mean())) if foreground.any() else float('nan'),
                     'mae_background': float(abs(err[background]).mean()) if background.any() else float('nan'),
                     'foreground_fraction': float(foreground.mean())})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    # A file is a patient only if your manifest establishes that correspondence.
    summary = {'n_files': len(rows), 'threshold': args.threshold, 'data_range': 1.0,
               'aggregation': 'unweighted per-file mean; patient macro only when one file = one patient',
               'note': 'GT-threshold mask is NOT an anatomical vessel annotation or the paper foreground protocol; empty masks produce NaN',
               'mean': {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'file'}}
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
