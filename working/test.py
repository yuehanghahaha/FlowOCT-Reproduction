"""Test script for FlowOCT: inference and save results.
Usage:
    python test.py --dataroot /root/autodl-tmp/FlowOCT/data/OCT_volume_6mm --name FlowOCT --model FlowOCT --dataset_name OCTA500 
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets.OCTA500_dataset import OCTA500Dataset
from models.FlowOCT_model import FlowOCT
from options.test_options import TestOptions


def main():
    opt = TestOptions().parse()
    device = torch.device(f'cuda:{opt.gpu_ids[0]}') if opt.gpu_ids else torch.device('cpu')

    # Dataset
    dataset = OCTA500Dataset(opt, phase='test')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # Model
    model = FlowOCT(opt)
    model.eval()

    # Output directory
    save_dir = os.path.join(opt.results_dir, opt.name, opt.epoch)
    os.makedirs(save_dir, exist_ok=True)

    for i, data in tqdm(enumerate(loader), total=len(loader), desc='Test'):
        real_A = data['A'].to(device, dtype=torch.float)
        real_B = data['B'].to(device, dtype=torch.float)
        fake_B = model.validation(condition=real_A, gt_image=None)
        
        real_A_np = real_A.cpu().numpy()[0, 0]
        real_B_np = real_B.cpu().numpy()[0, 0]
        fake_B_np = fake_B.cpu().numpy()[0, 0]
        
        filename = os.path.basename(data['A_paths'][0])
        save_path = os.path.join(save_dir, filename)
        np.savez_compressed(save_path, octa=fake_B_np, octa_gt=real_B_np, oct=real_A_np)

        print(f'Saved {save_path}')

    print(f'Done. Results saved to {save_dir}')


if __name__ == '__main__':
    main()