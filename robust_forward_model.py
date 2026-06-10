import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustForwardModel(nn.Module):
    def __init__(self, cond_dim, noise_std=0.001, hidden_dim=0,
                 w_min=0.1, w_max=1.5, l_upper=0.010, l_lower=0.004,
                 ema_decay=0.999):
        """
        RFM-LADS module for the Square low-dimensional task.

        The module applies dual-view observation perturbation, an asymmetric
        stop-gradient invariance loss, and an EMA-driven dynamic weight
        scheduler for the invariance term.
        """
        super().__init__()
        self.noise_std = noise_std
        self.w_min = w_min
        self.w_max = w_max
        self.l_upper = l_upper
        self.l_lower = l_lower
        self.ema_decay = ema_decay

        if hidden_dim > 0:
            self.encoder = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, cond_dim)
            )
        else:
            self.encoder = nn.Linear(cond_dim, cond_dim)
            nn.init.eye_(self.encoder.weight)
            nn.init.zeros_(self.encoder.bias)

        # EMA state used internally by LADS.
        self.register_buffer('ema_diff', torch.tensor(-1.0, dtype=torch.float32))

    def forward(self, obs_cond):
        """Training path: perturb observations, encode them, and compute raw invariance loss."""
        obs_1 = obs_cond + torch.randn_like(obs_cond) * self.noise_std
        obs_2 = obs_cond + torch.randn_like(obs_cond) * self.noise_std
        z1 = self.encoder(obs_1)
        z2 = self.encoder(obs_2)
        L_inv_raw = F.mse_loss(z1, z2.detach())
        
        # Return the first-view feature to the U-Net and the unweighted invariance loss.
        return z1, L_inv_raw

    def encode_for_inference(self, obs_cond):
        """Inference path: encode clean observations without perturbation noise."""
        return self.encoder(obs_cond)

    def get_dynamic_weight(self, current_diffusion_loss):
        """Update the EMA-smoothed diffusion loss and return the current dynamic weight."""
        if self.ema_diff.item() < 0:
            self.ema_diff.fill_(current_diffusion_loss)
        else:
            self.ema_diff.mul_(self.ema_decay).add_(
                current_diffusion_loss * (1.0 - self.ema_decay)
            )
            
        ema_val = self.ema_diff.item()

        # Boundary-based progress mapping followed by cosine smoothing.
        if ema_val >= self.l_upper:
            progress = 0.0
        elif ema_val <= self.l_lower:
            progress = 1.0
        else:
            progress = (self.l_upper - ema_val) / (self.l_upper - self.l_lower)

        cosine_factor = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.w_min + (self.w_max - self.w_min) * cosine_factor
