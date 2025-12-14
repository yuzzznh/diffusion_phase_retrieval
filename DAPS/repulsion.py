"""
Repulsion module for LatentDAPS: RLSD-style particle repulsion using DINO features.

This module implements SVGD-style repulsive gradients in DINO feature space,
ported from RLSD (Repulsive Latent Score Distillation) to work with LatentDAPS's
EDM sigma parameterization and x0-prediction.

Key components:
- DinoFeatureExtractor: Lazy-loaded DINO-ViT model for feature extraction
- compute_repulsion: SVGD repulsion gradient computation with RBF kernel
- RepulsionConfig: Configuration dataclass for repulsion parameters

References:
- RLSD: https://github.com/xxx/RLSD (rsd.py lines 123-165)
- SVGD: Stein Variational Gradient Descent
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any


@dataclass
class RepulsionConfig:
    """Configuration for particle repulsion."""
    scale: float = 0.0  # Initial repulsion strength (0 = disabled)
    sigma_break: float = 20.0  # Sigma threshold: repulsion active when sigma > sigma_break
    schedule: str = 'linear'  # Decay schedule: 'linear', 'cosine', 'constant'
    dino_model: str = 'dino_vits16'  # DINO model variant

    @classmethod
    def from_dict(cls, d: dict) -> 'RepulsionConfig':
        return cls(
            scale=d.get('repulsion_scale', 0.0),
            sigma_break=d.get('repulsion_sigma_break', 20.0),
            schedule=d.get('repulsion_schedule', 'linear'),
            dino_model=d.get('repulsion_dino_model', 'dino_vits16'),
        )


class DinoFeatureExtractor(nn.Module):
    """
    DINO-ViT feature extractor for computing repulsive gradients.

    Uses lazy loading to avoid loading the model unless repulsion is enabled.
    The model is frozen and set to eval mode for feature extraction.

    DINO expects:
    - Input: (B, 3, 224, 224) RGB images normalized with ImageNet stats
    - Output: (B, D) feature vectors where D depends on the model variant

    Note on resolution:
    - DAPS/LatentDAPS works with 256x256 images
    - RLSD works with 512x512 images
    - We resize to 224x224 for DINO regardless of input resolution
    """

    _instance: Optional['DinoFeatureExtractor'] = None

    def __init__(self, model_name: str = 'dino_vits16', device: Optional[torch.device] = None):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self._model = None
        self._loaded = False

        # ImageNet normalization for DINO
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @classmethod
    def get_instance(cls, model_name: str = 'dino_vits16', device: Optional[torch.device] = None) -> 'DinoFeatureExtractor':
        """Get or create singleton instance."""
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name, device)
        return cls._instance

    def _load_model(self):
        """Lazy load DINO model."""
        if not self._loaded:
            import sys
            import os
            import zipfile

            print(f"[Repulsion] Loading DINO model: {self.model_name}")

            # WORKAROUND: DINO's vision_transformer.py does `from utils import trunc_normal_`
            # which conflicts with DAPS's utils/ package.
            # Solution: Add DINO cache path FIRST in sys.path so DINO's utils.py takes precedence

            dino_cache_path = os.path.expanduser('~/.cache/torch/hub/facebookresearch_dino_main')
            hub_dir = os.path.expanduser('~/.cache/torch/hub')

            # Ensure DINO repo is downloaded first
            if not os.path.exists(dino_cache_path):
                # Download without executing (just to get the cache)
                zip_path = os.path.join(hub_dir, 'main.zip')
                torch.hub.download_url_to_file(
                    'https://github.com/facebookresearch/dino/zipball/main',
                    zip_path
                )
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(hub_dir)
                # Find the extracted folder name
                for name in os.listdir(hub_dir):
                    if name.startswith('facebookresearch-dino'):
                        os.rename(os.path.join(hub_dir, name), dino_cache_path)
                        break

            original_path = sys.path.copy()
            original_cwd = os.getcwd()

            try:
                # Remove any conflicting 'utils' module from sys.modules
                if 'utils' in sys.modules:
                    del sys.modules['utils']

                # Add DINO cache path at the START of sys.path
                # This ensures DINO's utils.py is found before DAPS's utils/
                sys.path.insert(0, dino_cache_path)

                # Also remove paths that might override DINO's utils
                sys.path = [dino_cache_path] + [
                    p for p in sys.path[1:]
                    if p and 'DAPS' not in p and 'diffusion_phase_retrieval' not in p
                ]

                # Change cwd to avoid local imports
                os.chdir(dino_cache_path)

                self._model = torch.hub.load('facebookresearch/dino:main', self.model_name, trust_repo=True)
            finally:
                # Restore original path, cwd, and clean up utils module
                sys.path = original_path
                os.chdir(original_cwd)
                # Remove DINO's utils from modules to avoid conflicts later
                if 'utils' in sys.modules and hasattr(sys.modules['utils'], 'trunc_normal_'):
                    del sys.modules['utils']

            self._model.eval()
            self._model.requires_grad_(False)
            if self.device is not None:
                self._model = self._model.to(self.device)
            self._loaded = True
            print(f"[Repulsion] DINO model loaded successfully")

    def to(self, device):
        """Move to device and update internal device."""
        self.device = device
        if self._loaded and self._model is not None:
            self._model = self._model.to(device)
        # Move normalization buffers
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for DINO input.

        Args:
            images: (B, 3, H, W) images in range [-1, 1] or [0, 1]

        Returns:
            (B, 3, 224, 224) normalized images for DINO
        """
        # Convert from [-1, 1] to [0, 1] if needed
        if images.min() < 0:
            images = (images + 1) / 2

        # Clamp to [0, 1]
        images = images.clamp(0, 1)

        # Resize to 224x224 for DINO
        if images.shape[-1] != 224 or images.shape[-2] != 224:
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

        # ImageNet normalization
        images = (images - self.mean.to(images.device)) / self.std.to(images.device)

        return images

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract DINO features from images.

        Args:
            images: (B, 3, H, W) images in range [-1, 1]

        Returns:
            (B, D) DINO feature vectors
        """
        self._load_model()

        # Preprocess
        images = self.preprocess(images)

        # Extract features (DINO outputs CLS token by default)
        features = self._model(images)

        return features


def compute_repulsion_gradient(
    latents: torch.Tensor,
    decode_fn,
    dino_extractor: DinoFeatureExtractor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute SVGD-style repulsive gradient in DINO feature space.

    This implements the core repulsion computation from RLSD:
    1. Decode latents to images
    2. Extract DINO features
    3. Compute pairwise distances
    4. Apply RBF kernel with median heuristic bandwidth
    5. Compute SVGD gradient
    6. Backprop to latent space using vector-Jacobian trick

    Args:
        latents: (B, C, H, W) latent tensors, requires_grad=True
        decode_fn: Function to decode latents to images (e.g., model.decode)
        dino_extractor: DinoFeatureExtractor instance
        eps: Small constant for numerical stability

    Returns:
        repulsion_grad: (B, C, H, W) repulsive gradient in latent space
        info: Dict with debugging info (distances, kernel values, etc.)

    Note:
        - Fixed N=2 bug from RLSD: uses max(log(N), eps) instead of log(N-1)
        - The gradient points in the direction that particles should move to repel
    """
    B = latents.shape[0]
    info = {}

    if B < 2:
        # No repulsion for single particle
        return torch.zeros_like(latents), {'skipped': True, 'reason': 'single_particle'}

    # Ensure latents require grad for backprop
    latents_for_grad = latents.clone()
    latents_for_grad.requires_grad_(True)

    # Decode latents to images
    # Note: decode_fn should handle the scaling internally
    images = decode_fn(latents_for_grad)

    # Extract DINO features
    dino_out = dino_extractor(images)  # (B, D)

    # Flatten features for pairwise computation
    features = dino_out.view(B, -1)  # (B, D)

    # Compute pairwise differences: diff[i, j] = features[i] - features[j]
    # Shape: (B, B, D)
    diff = features.unsqueeze(1) - features.unsqueeze(0)

    # Remove self-comparisons (diagonal) and reshape
    # For each particle i, get differences to all other particles
    # Shape after masking: (B, B-1, D)
    mask = ~torch.eye(B, dtype=bool, device=latents.device)
    diff = diff[mask].view(B, B - 1, -1)

    # Compute pairwise L2 distances
    # Shape: (B, B-1, 1)
    distance = torch.norm(diff, p=2, dim=-1, keepdim=True)

    # RBF kernel bandwidth using median heuristic
    # FIXED: Use max(log(N), eps) instead of log(N-1) to handle N=2 case
    log_N = max(np.log(B), eps)
    h_t = (distance.median(dim=1, keepdim=True)[0]) ** 2 / log_N
    h_t = h_t.clamp(min=eps)  # Prevent division by zero

    # RBF kernel weights: K(x_i, x_j) = exp(-||x_i - x_j||^2 / h)
    # Shape: (B, B-1, 1)
    weights = torch.exp(-(distance ** 2) / h_t)

    # SVGD gradient: grad_phi = sum_j K(x_i, x_j) * (x_i - x_j) / h
    # This is the "repulsive" term that pushes particles apart
    # Shape: (B, B-1, D) -> (B, D)
    grad_phi = 2 * weights * diff / h_t
    grad_phi = grad_phi.sum(dim=1)  # (B, D)

    # Vector-Jacobian trick to backprop from DINO features to latents
    # Instead of computing full Jacobian, we compute J^T @ grad_phi
    eval_sum = torch.sum(dino_out * grad_phi.detach())
    repulsion_grad = torch.autograd.grad(eval_sum, latents_for_grad, retain_graph=False)[0]

    # Normalize by kernel sum (SVGD normalization)
    K_sum = weights.sum(dim=1)  # (B, 1)
    K_sum = K_sum.view(B, 1, 1, 1).clamp(min=eps)
    repulsion_grad = repulsion_grad / K_sum

    # Collect debugging info
    weights_flat = weights.flatten()
    info = {
        'mean_pairwise_distance': distance.mean().item(),
        'median_pairwise_distance': distance.median().item(),
        'bandwidth_h': h_t.mean().item(),
        'mean_kernel_weight': weights.mean().item(),
        'repulsion_grad_norm': repulsion_grad.norm().item(),
        'num_particles': B,
        # Additional weight metrics for debugging kernel saturation
        'weights_mean': weights_flat.mean().item(),
        'weights_max': weights_flat.max().item(),
        'weights_min': weights_flat.min().item(),
        'weights_nonzero_frac': (weights_flat > 1e-6).float().mean().item(),
    }

    return repulsion_grad, info


def get_repulsion_scale(
    sigma: float,
    sigma_max: float,
    sigma_break: float,
    base_scale: float,
    schedule: str = 'linear',
) -> float:
    """
    Compute the repulsion scale based on current sigma and schedule.

    Repulsion is only active when sigma > sigma_break, and decays
    according to the schedule as sigma decreases.

    Args:
        sigma: Current noise level
        sigma_max: Maximum noise level (start of sampling)
        sigma_break: Threshold below which repulsion is disabled
        base_scale: Base repulsion strength (repulsion_scale config)
        schedule: Decay schedule ('linear', 'cosine', 'constant')

    Returns:
        Current repulsion scale (0 if sigma <= sigma_break)
    """
    if base_scale == 0 or sigma <= sigma_break:
        return 0.0

    # Compute progress within active range [sigma_break, sigma_max]
    # t = 1 at sigma_max, t = 0 at sigma_break
    t = (sigma - sigma_break) / (sigma_max - sigma_break + 1e-8)
    t = np.clip(t, 0, 1)

    if schedule == 'constant':
        return base_scale
    elif schedule == 'linear':
        # Linear decay: scale decreases as sigma decreases
        return base_scale * t
    elif schedule == 'cosine':
        # Cosine decay: smoother transition
        return base_scale * (1 + np.cos(np.pi * (1 - t))) / 2
    else:
        raise ValueError(f"Unknown repulsion schedule: {schedule}")


class RepulsionModule:
    """
    High-level repulsion manager for LatentDAPS.

    Manages DINO feature extractor, computes repulsion gradients,
    and tracks repulsion metrics for logging.

    Usage:
        repulsion_module = RepulsionModule(config, device)

        # In sampling loop:
        if repulsion_module.is_active(sigma, sigma_max):
            repulsion, info = repulsion_module.compute(latents, decode_fn, sigma, sigma_max)
            # Add repulsion to score or gradient
    """

    def __init__(self, config: RepulsionConfig, device: torch.device):
        self.config = config
        self.device = device
        self._dino = None

        # Metrics tracking
        self.step_metrics = []
        self.total_repulsion_time = 0.0

    @property
    def dino(self) -> DinoFeatureExtractor:
        """Lazy-load DINO extractor."""
        if self._dino is None:
            self._dino = DinoFeatureExtractor.get_instance(
                model_name=self.config.dino_model,
                device=self.device
            )
            self._dino.to(self.device)
        return self._dino

    def is_active(self, sigma: float, sigma_max: float) -> bool:
        """Check if repulsion should be computed at this sigma level."""
        if self.config.scale == 0:
            return False
        return sigma > self.config.sigma_break

    def get_scale(self, sigma: float, sigma_max: float) -> float:
        """Get current repulsion scale."""
        return get_repulsion_scale(
            sigma=sigma,
            sigma_max=sigma_max,
            sigma_break=self.config.sigma_break,
            base_scale=self.config.scale,
            schedule=self.config.schedule,
        )

    def compute(
        self,
        latents: torch.Tensor,
        decode_fn,
        sigma: float,
        sigma_max: float,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute scaled repulsion gradient.

        Args:
            latents: (B, C, H, W) latent tensors
            decode_fn: Function to decode latents to images
            sigma: Current noise level
            sigma_max: Maximum noise level

        Returns:
            scaled_repulsion: (B, C, H, W) scaled repulsion gradient
            info: Dict with metrics and debugging info
        """
        start_time = time.time()

        scale = self.get_scale(sigma, sigma_max)

        if scale == 0:
            return torch.zeros_like(latents), {
                'repulsion_active': False,
                'repulsion_scale': 0,
                'sigma': sigma,
            }

        repulsion_grad, grad_info = compute_repulsion_gradient(
            latents=latents,
            decode_fn=decode_fn,
            dino_extractor=self.dino,
        )

        scaled_repulsion = scale * repulsion_grad

        elapsed = time.time() - start_time
        self.total_repulsion_time += elapsed

        info = {
            'repulsion_active': True,
            'repulsion_scale': scale,
            'sigma': sigma,
            'repulsion_time_seconds': elapsed,
            **grad_info,
        }
        self.step_metrics.append(info)

        return scaled_repulsion, info

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Get summary metrics for logging."""
        if not self.step_metrics:
            return {'repulsion_enabled': self.config.scale > 0}

        active_steps = [m for m in self.step_metrics if m.get('repulsion_active', False)]

        return {
            'repulsion_enabled': self.config.scale > 0,
            'repulsion_total_steps': len(self.step_metrics),
            'repulsion_active_steps': len(active_steps),
            'repulsion_total_time_seconds': self.total_repulsion_time,
            'repulsion_mean_pairwise_distance': np.mean([m.get('mean_pairwise_distance', 0) for m in active_steps]) if active_steps else 0,
            'repulsion_config': {
                'scale': self.config.scale,
                'sigma_break': self.config.sigma_break,
                'schedule': self.config.schedule,
                'dino_model': self.config.dino_model,
            }
        }

    def reset_metrics(self):
        """Reset metrics for new sample."""
        self.step_metrics = []
        self.total_repulsion_time = 0.0
