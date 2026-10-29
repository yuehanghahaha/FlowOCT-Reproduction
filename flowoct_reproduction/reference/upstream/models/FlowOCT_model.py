"""FlowOCT model: flow matching with DWT for OCT-to-OCTA translation."""

import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.diffusion_model_unet import DiffusionModelUNet
from tqdm import tqdm

from util.DWT_IDWT_layer import DWT_2D, IDWT_2D
from util.auto_crop import auto_crop_octa_images


class FlowOCT(nn.Module):
    def name(self):
        return 'FlowOCTModel'

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if opt.gpu_ids else torch.device('cpu')
        self.isTrain = opt.isTrain
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    
        # Generator: 4 channels = DWT decomposition (LL, LH, HL, HH)
        self.netG = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=4,
            out_channels=4,
        ).to(self.device)

        # DWT/IDWT layers
        self.dwt_2d = DWT_2D('haar')
        self.idwt_2d = IDWT_2D('haar')
        
        # Projection head for contrastive learning
        self.contrast_projection_head = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
        ).to(self.device)

        # Loss functions
        self.criterionL1 = nn.L1Loss()
        self.criterionL2 = nn.MSELoss()

        # Loss weights
        self.vweight = 10.0
        self.L1weight = 0.5
        self.contrastweight = 0.005
        self.projweight = 0.1

        # Optimizer
        if self.isTrain:
            params = list(self.netG.parameters()) + list(self.contrast_projection_head.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta, 0.999))

        # Load checkpoint if resuming
        if not self.isTrain or opt.continue_train:
            self._load_checkpoint(opt.epoch)
        
        self.generation_steps = 3

    def _load_checkpoint(self, epoch):
        """Load saved model weights if resuming training."""
        self.load(epoch)
        print(f'Loaded checkpoint from epoch {epoch}')

    def set_input(self, input):
        """Set input data for training."""
        self.real_A = input['A'].to(self.device,dtype=torch.float)
        self.real_B = input['B'].to(self.device,dtype=torch.float)

    # ---- Flow matching core ----
    def euler_step(self, x_t, t_start, t_end):
        """Single Euler step for flow matching ODE."""
        delta_t = (t_end - t_start).view(-1, 1, 1, 1)

        LL, LH, HL, HH = self.dwt_2d(x_t)
        v_dwt = self.netG(torch.cat([LL, LH, HL, HH], dim=1), t_start)

        v = self.idwt_2d(v_dwt[:, 0:1], v_dwt[:, 1:2], v_dwt[:, 2:3], v_dwt[:, 3:4])
        return x_t + delta_t * v

    @torch.no_grad()
    def generate(self, condition, gen_steps=None):
        """Generate OCTA from OCT condition via flow ODE."""
        if gen_steps is None:
            gen_steps = self.generation_steps

        x = condition
        time_steps = torch.linspace(0.0, 1.0, gen_steps + 1, device=self.device)
        batch_size = condition.shape[0]

        for i in range(gen_steps):
            t_start = time_steps[i].expand(batch_size)
            t_end = time_steps[i + 1].expand(batch_size)
            x = self.euler_step(x, t_start, t_end)
            x = torch.clamp(x, 0, 1)
            
        return x

    # ---- Training ----
    
    def forward(self):
        """Training step with flow matching, contrastive, and projection losses."""
        B, C, D, H, W = self.real_A.shape
        
        # Sample random 2D slices along depth
        num_slices = max(1, D // 4)
        indices = torch.randperm(D, device=self.device)[:num_slices]
        
        oct_slices = self.real_A[:, :, indices, :, :]   # (B, C, N, H, W)
        octa_slices = self.real_B[:, :, indices, :, :]  # (B, C, N, H, W)
        
        # Flatten: (B, C, N, H, W) -> (B*N, C, H, W)
        oct_flat = oct_slices.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        octa_flat = octa_slices.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)

        # Flow matching: x_t = (1-t)*x_0 + t*x_1, target velocity = x_1 - x_0
        t = torch.rand(oct_flat.shape[0], device=self.device)
        t_img = t.view(-1, 1, 1, 1)

        x_t = (1 - t_img) * oct_flat + t_img * octa_flat
        dx_t = octa_flat - oct_flat
        

        # DWT forward
        LL, LH, HL, HH = self.dwt_2d(x_t)
        x_t_dwt = torch.cat([LL, LH, HL, HH], dim=1)

        pred_dx_dwt = self.netG(x_t_dwt, t)
  
        # DWT inverse
        pred_dx = self.idwt_2d(
            pred_dx_dwt[:, 0:1], pred_dx_dwt[:, 1:2],
            pred_dx_dwt[:, 2:3], pred_dx_dwt[:, 3:4],
        )
        
        # Predicted OCTA
        pred_octa_flat = x_t + (1 - t_img) * pred_dx
        pred_octa_flat = torch.clamp(pred_octa_flat, 0, 1)
        
        # ---- Compute losses ----
        self.optimizer_G.zero_grad()
        
        # Velocity loss with auto-crop
        self.vloss = self._cropped_l2_loss(pred_dx, dx_t)
        
        # L1 reconstruction loss with auto-crop
        self.L1loss = self._cropped_l1_loss(pred_octa_flat, octa_flat)

        # Contrastive loss
        feat_fake = self.contrast_projection_head(pred_octa_flat)
        feat_real = self.contrast_projection_head(octa_flat)
        self.contrastloss = self._contrastive_loss(feat_fake, feat_real)


        # Projection consistency loss (max projection) 
        real_proj = octa_flat.max(dim=0)[0]  # (C, H, W)
        fake_proj = pred_octa_flat.max(dim=0)[0]
        _, _, bbox = auto_crop_octa_images(
            fake_proj.squeeze(0).detach().cpu().numpy(),
            real_proj.squeeze(0).detach().cpu().numpy(),
        )
        xm, ym, xM, yM = bbox['expanded_bbox']
        self.proj_loss = self.criterionL1(
            fake_proj[ym:yM+1, xm:xM+1],
            real_proj[ym:yM+1, xm:xM+1],
        )

        # Total loss
        self.loss = (
            self.vweight * self.vloss +
            self.L1weight * self.L1loss +
            self.contrastweight * self.contrastloss +
            self.projweight * self.proj_loss
        )
        
        self.loss.backward()
        self.optimizer_G.step()
    
    def _cropped_l2_loss(self, pred, target):
        """L2 loss on auto-cropped region."""
        loss = 0.0
        for i in range(pred.shape[0]):
            _, _, bbox = auto_crop_octa_images(
                pred[i].detach().cpu().numpy().squeeze(0),
                target[i].detach().cpu().numpy().squeeze(0),
            )
            xm, ym, xM, yM = bbox['expanded_bbox']
            loss += self.criterionL2(
                pred[i, :, ym:yM+1, xm:xM+1],
                target[i, :, ym:yM+1, xm:xM+1],
            )
        return loss / pred.shape[0]

    def _cropped_l1_loss(self, pred, target):
        """L1 loss on auto-cropped region."""
        loss = 0.0
        for i in range(pred.shape[0]):
            _, _, bbox = auto_crop_octa_images(
                pred[i].detach().cpu().numpy().squeeze(0),
                target[i].detach().cpu().numpy().squeeze(0),
            )
            xm, ym, xM, yM = bbox['expanded_bbox']
            loss += self.criterionL1(
                pred[i, :, ym:yM+1, xm:xM+1],
                target[i, :, ym:yM+1, xm:xM+1],
            )
        return loss / pred.shape[0]

    def _contrastive_loss(self, feat_fake, feat_real, temperature=0.1):
        """InfoNCE contrastive loss (slice-level)."""
        D = feat_fake.shape[0]
        features = F.normalize(torch.cat([feat_fake, feat_real], dim=0), dim=1)

        sim = features @ features.T  # (2D, 2D)

        # Positive mask: (i, i+D) and (i+D, i)
        pos_mask = torch.zeros(2 * D, 2 * D, dtype=torch.bool, device=features.device)
        pos_mask[torch.arange(D), torch.arange(D) + D] = True
        pos_mask[torch.arange(D) + D, torch.arange(D)] = True

        # Negative mask: not self, not positive
        neg_mask = ~(torch.eye(2 * D, dtype=torch.bool, device=features.device) | pos_mask)

        pos_sim = sim[pos_mask].view(2 * D, 1)
        neg_sim = sim[neg_mask].view(2 * D, -1)

        logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature
        labels = torch.zeros(2 * D, dtype=torch.long, device=features.device)

        return F.cross_entropy(logits, labels)


    # ---- Inference ----
    
    def _generate_3d(self, condition, chunk_size=32):
        """Generate full 3D volume by processing slices in chunks."""
        B, C, D, H, W = condition.shape
        flat = condition.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)

        outputs = []
        for start in range(0, flat.shape[0], chunk_size):
            chunk = flat[start:start + chunk_size]
            outputs.append(self.generate(chunk))

        fake_flat = torch.cat(outputs, dim=0)
        fake_flat = torch.clamp(fake_flat, 0, 1)

        return fake_flat.reshape(B, D, C, H, W).permute(0, 2, 1, 3, 4)

    def validation(self, condition, gt_image, chunk_size=32):
        """Validation: generate OCTA and return volume."""
        with torch.no_grad():
            return self._generate_3d(condition.to(self.device), chunk_size)

    def test(self, chunk_size=32):
        """Test: generate and store fake_B."""
        with torch.no_grad():
            self.fake_B = self._generate_3d(self.real_A, chunk_size)
            return self.fake_B

    # ---- Utilities ----
    def get_current_loss(self):
        return OrderedDict([('vloss', self.vloss.item()),
                            ('L1loss', self.L1loss.item()),
                            ('contrastloss', self.contrastloss.item()),
                            ('proj_loss', self.proj_loss.item()),
                           ])

    def save(self, label):
        """Save checkpoint."""
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(self.netG.state_dict(), os.path.join(self.save_dir, f'{label}_net_G.pth'))
        torch.save(self.contrast_projection_head.state_dict(), os.path.join(self.save_dir, f'{label}_net_Contrast.pth'))
    
    def load(self, label):
        """Load checkpoint."""
        self.netG.load_state_dict(torch.load(os.path.join(self.save_dir, f'{label}_net_G.pth'), map_location=self.device))
        self.contrast_projection_head.load_state_dict(torch.load(os.path.join(self.save_dir, f'{label}_net_Contrast.pth'), map_location=self.device))
