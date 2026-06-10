import math
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

from diffusion_policy.model.common.robust_forward_model import RobustForwardModel
import wandb

class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        # Enable the RFM module and the LADS scheduling engine.
        self.use_rfm = True
        self._log_interval = 500
        self._step_counter = 0

        if self.use_rfm:
            cond_dim = obs_dim * n_obs_steps if obs_as_global_cond else obs_dim
            # Square-task RFM-LADS hyperparameters are encapsulated in the RFM module.
            self.rfm = RobustForwardModel(
                cond_dim=cond_dim, 
                noise_std=0.001, 
                hidden_dim=0,
                w_min=0.1,       # Early-stage low regularization.
                w_max=1.5,       # Late-stage regularization upper bound.
                l_upper=0.010,   # Upper loss threshold for progress mapping.
                l_lower=0.004    # Lower loss threshold for progress mapping.
            )
        else:
            self.rfm = None

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        trajectory[condition_mask] = condition_data[condition_mask]        
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            obs_flat = nobs[:,:To].reshape(nobs.shape[0], -1)
            # Inference uses the encoder without perturbation noise.
            if self.rfm is not None and self.use_rfm:
                global_cond = self.rfm.encode_for_inference(obs_flat)
            else:
                global_cond = obs_flat
            
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch, global_step=0):
        # 1. Normalize inputs using the policy normalizer.
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']
        B = action.shape[0]

        # 2. Build trajectories and conditioning inputs.
        obs_cond = obs[:, :self.n_obs_steps, :].reshape(B, -1)
        local_cond = None
        trajectory = action

        if self.obs_as_local_cond:
            local_cond = obs
            local_cond[:, self.n_obs_steps:, :] = 0
        elif self.obs_as_global_cond:
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To - 1 if self.oa_step_convention else To
                end = start + self.n_action_steps
                trajectory = action[:, start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # 3. Pass observations through RFM to obtain robust features and the raw invariance loss.
        if self.rfm is not None and self.use_rfm:
            global_cond, L_inv_raw = self.rfm(obs_cond)
        else:
            global_cond = obs_cond
            L_inv_raw = None

        # 4. Generate masks, add diffusion noise, and predict the denoising target.
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (B,), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # The U-Net receives the RFM-encoded global condition, so gradients can update the encoder.
        pred = self.model(noisy_trajectory, timesteps, 
                          local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        target = noise if pred_type == 'epsilon' else trajectory

        # 5. Compute the main diffusion denoising loss.
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        diffusion_loss = loss.mean()

        # 6. LADS: compute the dynamic invariance weight and combine losses.
        if L_inv_raw is not None:
            # Detach the diffusion loss before updating the EMA scheduler state.
            curr_loss_val = diffusion_loss.detach().item()
            
            # Query the current dynamic weight from the RFM-LADS scheduler.
            dynamic_weight = self.rfm.get_dynamic_weight(curr_loss_val)

            # Combine the diffusion loss and weighted invariance loss.
            weighted_L_inv = dynamic_weight * L_inv_raw
            final_loss = diffusion_loss + weighted_L_inv

            # Low-frequency diagnostic logging.
            self._step_counter += 1
            if self._step_counter % self._log_interval == 0:
                wandb.log({
                    'rfm/z_std': global_cond.std(dim=0).mean().item(),
                    'rfm/L_inv_raw': L_inv_raw.item(),
                    'rfm/diffusion_loss': curr_loss_val,
                    'rfm/ema_diff_loss': self.rfm.ema_diff.item(),
                    'rfm/current_weight': dynamic_weight,
                    'rfm/total_loss': final_loss.item(),
                    'rfm/L_inv_ratio': weighted_L_inv.item() / (curr_loss_val + 1e-8)
                })
        else:
            final_loss = diffusion_loss

        return final_loss
