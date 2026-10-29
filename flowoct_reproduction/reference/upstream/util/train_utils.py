"""Training and validation loop utilities for FlowOCT."""

import time
import torch
from tqdm import tqdm

from util.util import get_log_path, log_loss, log_metrics, compute_mae, calculate_psnr_3d, calculate_ssim_3d
    
def run_training_epoch(model, loader, epoch, opt):
    log_path = get_log_path(opt)
    iter_start = time.time()
    epoch_iter = 0

    pbar = tqdm(enumerate(loader), total=len(loader),
                desc=f'Epoch {epoch}/{opt.n_epochs} [Train]')

    for _, data in pbar:
        epoch_iter += opt.batch_size
        model.set_input(data)
        model.forward()

        if epoch_iter % opt.print_freq == 0:
            loss = model.get_current_loss()
            step_time = (time.time() - iter_start) / opt.batch_size
            log_loss(log_path, epoch, epoch_iter, loss, step_time, phase='Train')


def run_validation(model, val_loader, epoch, opt, device, global_best_psnr):
    log_path = get_log_path(opt)
    val_start = time.time()

    mae_total = psnr_total = ssim_total = 0.0
    num = 0

    pbar = tqdm(enumerate(val_loader), total=len(val_loader),
                desc=f'Epoch {epoch}/{opt.n_epochs} [Val]')

    with torch.no_grad():
        for _, data in pbar:
            real_A = data['A'].to(device, dtype=torch.float)
            real_B = data['B'].to(device, dtype=torch.float)

            fake_B = model.validation(condition=real_A, gt_image=real_B)

            real_np = real_B.detach().cpu().numpy()
            fake_np = fake_B.detach().cpu().numpy()

            mae_total += compute_mae(fake_np, real_np)

            real_3d = real_np[0, 0]
            fake_3d = fake_np[0, 0]

            psnr_val = calculate_psnr_3d(real_3d, fake_3d)
            ssim_val = calculate_ssim_3d(real_3d, fake_3d, window_size=7)

            psnr_total += psnr_val
            ssim_total += ssim_val
            num += 1

    avg_mae = mae_total / num
    avg_psnr = psnr_total / num
    avg_ssim = ssim_total / num
    val_time = time.time() - val_start

    log_metrics(log_path, epoch, avg_mae, avg_psnr, avg_ssim, val_time)
    print(f'[Compute Cost] Epoch {epoch}: Validation time = {val_time:.2f}s')

    if epoch % opt.save_epoch_freq == 0:
        name = f'current_epoch_{epoch:03d}_psnr_{avg_psnr:.4f}_ssim_{avg_ssim:.4f}'
        print(f'Saving checkpoint: {name}')
        model.save(name)

    if avg_psnr > global_best_psnr:
        global_best_psnr = avg_psnr
        name = f'best_epoch_{epoch:03d}_psnr_{avg_psnr:.4f}_ssim_{avg_ssim:.4f}'
        print(f'Saving best model: {name}')
        model.save(name)

    return global_best_psnr