"""
Training script for FlowOCT: 3D OCT image-to-OCTA translation.
Usage:
    python train.py --dataroot /root/autodl-tmp/FlowOCT/data/OCT_volume_6mm --name FlowOCT --model FlowOCT --dataset_name OCTA500 
"""

import time
import torch

from options.train_options import TrainOptions
from datasets.OCTA500_dataset import OCTA500Dataset
from torch.utils.data import DataLoader
from models.FlowOCT_model import FlowOCT
from util.train_utils import run_training_epoch, run_validation


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """Main training loop for FlowOCT."""
    # Parse options
    opt = TrainOptions().parse()
    device = torch.device(f'cuda:{opt.gpu_ids[0]}') if opt.gpu_ids else torch.device('cpu')

    # Create datasets
    train_dataset = OCTA500Dataset(opt, phase='train')
    val_dataset = OCTA500Dataset(opt, phase='validation')
    print(f'Training images = {len(train_dataset)}')
    print(f'Validation images = {len(val_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                          num_workers=opt.num_threads, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                        num_workers=opt.num_threads)

    # Create and setup FlowOCT model
    model = FlowOCT(opt)

    # Report model complexity
    total_params = count_parameters(model.netG)
    print(f'[Compute Cost] FlowOCT model parameters: {total_params / 1e6:.2f}M ({total_params:,})')

    # Training state
    global_best_psnr = 0.0

    # Training loop
    for epoch in range(opt.epoch_count, opt.n_epochs + 1):
        # Training phase
        run_training_epoch(model, train_loader, epoch, opt)

        # Validation phase
        if epoch % opt.val_epoch_freq == 0:
            global_best_psnr = run_validation(model, val_loader, epoch, opt, device,global_best_psnr)

        print(f'End of epoch {epoch} / {opt.n_epochs}')

if __name__ == '__main__':
    main()