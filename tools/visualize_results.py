"""Visualize index planes and absolute error. Labels do not claim anatomical orientation."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = [args.input] if args.input.is_file() else sorted(args.input.rglob('*.npz'))
    if not files:
        raise ValueError('No result files')
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            a, pred, gt = data['oct'], data['octa'], data['octa_gt']
        fig, axes = plt.subplots(3, 4, figsize=(11, 8), constrained_layout=True)
        for row in range(3):
            for col, (volume, label) in enumerate([(a, 'OCT'), (pred, 'Predicted OCTA'), (gt, 'Target OCTA'), (abs(pred-gt), 'Absolute error')]):
                img = np.take(volume, volume.shape[row] // 2, axis=row)
                axes[row, col].imshow(img, cmap='magma' if col == 3 else 'gray', vmin=0, vmax=1)
                axes[row, col].set_title(f'Index axis {row}: {label}', fontsize=9)
                axes[row, col].axis('off')
        fig.suptitle('Synthetic demonstration only' if path.name.startswith('synthetic') else path.name)
        fig.savefig(args.output / (path.stem + '.png'), dpi=130)
        plt.close(fig)


if __name__ == '__main__':
    main()
