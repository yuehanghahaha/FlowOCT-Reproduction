"""Dataset class for FlowOCT: loading paired OCT and OCTA 3D volumes."""


import os
import random
import numpy as np
import torch
import torchio as tio

def _make_dataset(directory, max_size=float("inf")):
    """Collect file paths from directory and subdirectories."""
    paths = []
    for root, _, fnames in sorted(os.walk(directory)):
        for fname in fnames:
            if fname.endswith('.npz'):
                paths.append(os.path.join(root, fname))
    return paths[:min(max_size, len(paths))]


class OCTA500Dataset(torch.utils.data.Dataset):
    """Paired OCT-to-OCTA 3D dataset.

    Each .npz file contains 'oct' and 'octa' arrays.
    """

    def __init__(self, opt, phase):
        self.opt = opt
        self.phase = phase

        data_dir = os.path.join(opt.dataroot, phase)
        self.paths = sorted(_make_dataset(data_dir))
        print(f"[Dataset] {phase}: {len(self.paths)} volumes")

        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc


    def _get_crop_params(self, shape):
        """Random crop (train) or center crop (val/test)."""
        _, d, h, w = shape  # (C, D, H, W)

        if self.phase == 'train':
            d_start = random.randint(0, max(0, d - self.opt.crop_size))
            h_start = random.randint(0, max(0, h - self.opt.crop_size))
            w_start = random.randint(0, max(0, w - self.opt.crop_size))
            flip = random.random() > 0.5
        else:
            d_start = (d - self.opt.crop_size) // 2
            h_start = (h - self.opt.crop_size) // 2
            w_start = (w - self.opt.crop_size) // 2
            flip = False

        return {
            'crop': (d_start, d_start + self.opt.crop_size,
                     h_start, h_start + self.opt.crop_size,
                     w_start, w_start + self.opt.crop_size),
            'flip': flip,
        }
        
    def __getitem__(self, index):        
        path = self.paths[index]

        data = np.load(path) 

        A = np.expand_dims(data['oct'], axis=0) # [1,400,400,640]
        B = np.expand_dims(data['octa'], axis=0)

        # Resize
        A = self._resize(A)
        B = self._resize(B)

        # Crop & flip
        params = self._get_crop_params(A.shape)
        c = params['crop']
        A = A[:, c[0]:c[1], c[2]:c[3], c[4]:c[5]]
        B = B[:, c[0]:c[1], c[2]:c[3], c[4]:c[5]]

        if params['flip']:
            A = np.flip(A, axis=-1).copy()
            B = np.flip(B, axis=-1).copy()
        
        # min-max normalize to [0, 1]
        A = (A - A.min()) / (A.max() - A.min())
        B = (B - B.min()) / (B.max() - B.min())

        return {
            'A': torch.from_numpy(A),
            'B': torch.from_numpy(B),
            'A_paths': path,
            'B_paths': path,
        }

    def _resize(self, x):
        """Resize volume to (load_size, load_size, crop_size)."""
        transform = tio.Resize(
            (self.opt.load_size, self.opt.load_size, self.opt.crop_size)
        )
        return transform(x)

    def __len__(self):
        return len(self.paths)




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--input_nc', type=int, default=1)
    parser.add_argument('--output_nc', type=int, default=1)
    parser.add_argument('--load_size', type=int, default=400)
    parser.add_argument('--crop_size', type=int, default=256)
    opt = parser.parse_args()

    dataset = OCTA500Dataset(opt, opt.phase)
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"A shape: {sample['A'].shape}, dtype: {sample['A'].dtype}")
    print(f"B shape: {sample['B'].shape}, dtype: {sample['B'].dtype}")
    print(f"A min/max: {sample['A'].min():.4f} / {sample['A'].max():.4f}")
    print(f"B min/max: {sample['B'].min():.4f} / {sample['B'].max():.4f}")

#python OCTA500_dataset.py --dataroot /root/autodl-tmp/cwdm-main-oct/datasets/OCT_volume_6mm_denoise --phase train