"""Utility functions for FlowOCT."""

import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from util.auto_crop import auto_crop_octa_images


# ---- Logging ----

def get_log_path(opt):
    """Create log directory and return log file path."""
    log_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, 'loss_log.txt')


def log_loss(log_path, epoch, iters, loss, step_time, phase='Train'):
    """Log training losses to console and file."""
    message = f'[{phase}] epoch: {epoch}, iters: {iters}, time: {step_time:.3f}s'
    for k, v in loss.items():
        message += f', {k}: {v:.5f}'

    print(message)
    with open(log_path, 'a') as f:
        f.write(message + '\n')


def log_metrics(log_path, epoch, mae, psnr, ssim, val_time):
    """Log validation metrics to console and file."""
    message = (
        f'[Val] epoch: {epoch}, time: {val_time:.2f}s, '
        f'mae: {mae:.5f}, psnr: {psnr:.3f}, ssim: {ssim:.4f}'
    )
    print(message)
    with open(log_path, 'a') as f:
        f.write(message + '\n')


# ---- Metrics ----

def compute_mae(fake, real):
    """Mean Absolute Error."""
    return float(np.mean(np.abs(fake - real)))


def calculate_psnr_3d(real, fake):
    """Calculate PSNR for 3D volumes."""

    data_range = real.max() - real.min()
    psnr_3d = peak_signal_noise_ratio(real, fake, data_range=data_range)
    return psnr_3d


def calculate_ssim_3d(real, fake, window_size=7):
    """Calculate SSIM for 3D volumes."""

    data_range = real.max() - real.min()
    ssim_3d = structural_similarity(real, fake, data_range=data_range)
    return ssim_3d





