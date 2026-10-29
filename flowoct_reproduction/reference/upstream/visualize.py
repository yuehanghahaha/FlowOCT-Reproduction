"""Visualize OCT slices, predicted OCTA, and ground truth OCTA from three views."""

import os
import numpy as np
import matplotlib.pyplot as plt

RESULT_DIR = ''
OUTPUT_DIR = os.path.join(RESULT_DIR, 'slices')
NUM_SAMPLES = 5


def load_volume(path):
    data = np.load(path)
    return data['oct'], data['octa'], data['octa_gt']


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    names = sorted([f for f in os.listdir(RESULT_DIR) if f.endswith('.npz')])[:NUM_SAMPLES]

    for name in names:
        oct_gt, octa_pred, octa_gt = load_volume(os.path.join(RESULT_DIR, name))
        D, H, W = oct_gt.shape

        slices = {
            'Axial':    (oct_gt[D // 2, :, :],   octa_pred[D // 2, :, :],   octa_gt[D // 2, :, :]),
            'Coronal':  (oct_gt[:, H // 2, :],   octa_pred[:, H // 2, :],   octa_gt[:, H // 2, :]),
            'Sagittal': (oct_gt[:, :, W // 2],   octa_pred[:, :, W // 2],   octa_gt[:, :, W // 2]),
        }

        fig, axes = plt.subplots(3, 3, figsize=(9, 9))

        titles = ['OCT', 'OCTA (pred)', 'OCTA (GT)']

        for row, (view, (s_oct, s_pred, s_gt)) in enumerate(slices.items()):
            for col, (img, title) in enumerate(zip([s_oct, s_pred, s_gt], titles)):
                ax = axes[row, col]
                ax.imshow(img.T if view != 'Axial' else img, cmap='gray', origin='lower')
                if row == 0:
                    ax.set_title(title, fontsize=8)
                if col == 0:
                    ax.set_ylabel(view, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])

        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        save_path = os.path.join(OUTPUT_DIR, f'{os.path.splitext(name)[0]}_3views.png')
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f'Saved {save_path}')


if __name__ == '__main__':
    main()