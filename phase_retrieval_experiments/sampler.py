import time
import tqdm
import torch
import numpy as np
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from cores.trajectory import Trajectory
from cores.scheduler import get_diffusion_scheduler, DiffusionPFODE
from cores.mcmc import MCMCSampler
from forward_operator import LatentWrapper
from utils import mark_step
from repulsion import RepulsionModule, RepulsionConfig


def get_sampler(**kwargs):
    latent = kwargs['latent']
    kwargs.pop('latent')
    if latent:
        return LatentDAPS(**kwargs)
    return DAPS(**kwargs)


class DAPS(nn.Module):
    """
    Decoupled Annealing Posterior Sampling (DAPS) implementation.

    Combines diffusion models and MCMC updates for posterior sampling from noisy measurements.
    """

    def __init__(self, annealing_scheduler_config, diffusion_scheduler_config, mcmc_sampler_config,
                 repulsion_scale=0.0, pruning_step=-1, hard_data_consistency=-1, use_tpu=False):
        """
        Initializes the DAPS sampler with the provided scheduler and sampler configurations.

        Args:
            annealing_scheduler_config (dict): Configuration for annealing scheduler.
            diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
            mcmc_sampler_config (dict): Configuration for MCMC sampler.
            repulsion_scale (float): Initial strength of particle repulsion. 0.0 = independent (DAPS baseline).
            pruning_step (int): Timestep to perform pruning. -1 = no pruning.
            hard_data_consistency (int): Timestep to start latent optimization. -1 = no optimization.
            use_tpu (bool): True면 TPU 사용 (mark_step 호출), False면 CUDA 사용.
        """
        super().__init__()
        annealing_scheduler_config, diffusion_scheduler_config = self._check(annealing_scheduler_config,
                                                                             diffusion_scheduler_config)
        self.annealing_scheduler = get_diffusion_scheduler(**annealing_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.mcmc_sampler = MCMCSampler(**mcmc_sampler_config)

        # 실험 1~5용 Flag
        self.repulsion_scale = repulsion_scale
        self.pruning_step = pruning_step
        self.hard_data_consistency = hard_data_consistency

        # TPU/CUDA Flag
        self.use_tpu = use_tpu

        # Gradient 필요 여부 (실험 0에서는 False)
        self.needs_grad = (repulsion_scale > 0) or (hard_data_consistency == 1)

    def sample(self, model, x_start, operator, measurement, evaluator=None, record=False, verbose=False, **kwargs):
        """
        Performs sampling using the DAPS method.

        Args:
            model (nn.Module): Diffusion model.
            x_start (torch.Tensor): Initial tensor/state.
            operator (nn.Module): Measurement operator.
            measurement (torch.Tensor): Observed measurement tensor.
            evaluator (Evaluator, optional): Evaluator for performance metrics.
            record (bool, optional): If True, records the sampling trajectory.
            verbose (bool, optional): Enables progress bar and logs.
            **kwargs:
                gt (torch.Tensor, optional): Ground truth data for evaluation.

        Returns:
            torch.Tensor: Final sampled tensor/state.
        """
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.trange(self.annealing_scheduler.num_steps - 1) if verbose else range(self.annealing_scheduler.num_steps - 1)
        xt = x_start
        for step in pbar:
            sigma = self.annealing_scheduler.sigma_steps[step]
            # 1. reverse diffusion
            with torch.no_grad():
                diffusion_scheduler = get_diffusion_scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
                sampler = DiffusionPFODE(model, diffusion_scheduler, solver='euler')
                x0hat = sampler.sample(xt)

            # 2. MCMC update
            x0y = self.mcmc_sampler.sample(xt, model, x0hat, operator, measurement, sigma, step / self.annealing_scheduler.num_steps)

            # 3. forward diffusion
            if step != self.annealing_scheduler.num_steps - 1:
                xt = x0y + torch.randn_like(x0y) * self.annealing_scheduler.sigma_steps[step + 1]
            else:
                xt = x0y

            # 4. evaluation
            x0hat_results = x0y_results = {}
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    x0hat_results = evaluator(gt, measurement, x0hat)
                    x0y_results = evaluator(gt, measurement, x0y)

                # record
                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        'x0hat' + '_' + main_eval_fn_name: f"{x0hat_results[main_eval_fn_name].item():.2f}",
                        'x0y' + '_' + main_eval_fn_name: f"{x0y_results[main_eval_fn_name].item():.2f}",
                    })
            if record:
                self._record(xt, x0y, x0hat, sigma, x0hat_results, x0y_results)

            # TPU: 매 step 끝에서 mark_step() 호출 (lazy execution 그래프 실행)
            mark_step(self.use_tpu)
        return xt

    def _record(self, xt, x0y, x0hat, sigma, x0hat_results, x0y_results):
        """Records the intermediate states during sampling."""

        self.trajectory.add_tensor(f'xt', xt)
        self.trajectory.add_tensor(f'x0y', x0y)
        self.trajectory.add_tensor(f'x0hat', x0hat)
        self.trajectory.add_value(f'sigma', sigma)
        for name in x0hat_results.keys():
            self.trajectory.add_value(f'x0hat_{name}', x0hat_results[name])
        for name in x0y_results.keys():
            self.trajectory.add_value(f'x0y_{name}', x0y_results[name])

    def _check(self, annealing_scheduler_config, diffusion_scheduler_config):
        """Checks and updates the configurations for the schedulers."""

        # sigma_max of diffusion scheduler change each step
        if 'sigma_max' in diffusion_scheduler_config:
            diffusion_scheduler_config.pop('sigma_max')

        return annealing_scheduler_config, diffusion_scheduler_config

    def get_start(self, batch_size, model):
        """
        Generates initial random state tensors from the Gaussian prior.

        Args:
            batch_size (int): Number of initial states to generate.
            model (nn.Module): Diffusion or latent diffusion model.

        Returns:
            torch.Tensor: Random initial tensor.
        """
        device = next(model.parameters()).device
        in_shape = model.get_in_shape()
        x_start = torch.randn(batch_size, *in_shape, device=device) * self.annealing_scheduler.get_prior_sigma()
        return x_start


class LatentDAPS(DAPS):
    """
    Latent Decoupled Annealing Posterior Sampling (LatentDAPS).

    Implements posterior sampling using a latent diffusion model combined with MCMC updates.

    Supports particle repulsion (Exp 1/3) via score-level injection:
    - When repulsion_scale > 0, particles repel each other in DINO feature space
    - Repulsion is active when sigma > sigma_break and decays according to schedule
    """

    def __init__(self, annealing_scheduler_config, diffusion_scheduler_config, mcmc_sampler_config,
                 repulsion_scale=0.0, repulsion_sigma_break=20.0, repulsion_schedule='linear',
                 repulsion_dino_model='dino_vits16',
                 pruning_step=-1, hard_data_consistency=-1, use_tpu=False):
        """
        Initialize LatentDAPS with repulsion support.

        Args:
            annealing_scheduler_config: Config for annealing scheduler
            diffusion_scheduler_config: Config for diffusion scheduler
            mcmc_sampler_config: Config for MCMC sampler
            repulsion_scale: Initial repulsion strength (0 = disabled, DAPS baseline)
            repulsion_sigma_break: Sigma threshold below which repulsion is disabled
            repulsion_schedule: Decay schedule ('linear', 'cosine', 'constant')
            repulsion_dino_model: DINO model variant for feature extraction
            pruning_step: Timestep for pruning (-1 = disabled)
            hard_data_consistency: Timestep to start optimization (-1 = disabled)
            use_tpu: Whether to use TPU
        """
        super().__init__(
            annealing_scheduler_config=annealing_scheduler_config,
            diffusion_scheduler_config=diffusion_scheduler_config,
            mcmc_sampler_config=mcmc_sampler_config,
            repulsion_scale=repulsion_scale,
            pruning_step=pruning_step,
            hard_data_consistency=hard_data_consistency,
            use_tpu=use_tpu,
        )

        # Repulsion configuration (Exp 1/3)
        self.repulsion_config = RepulsionConfig(
            scale=repulsion_scale,
            sigma_break=repulsion_sigma_break,
            schedule=repulsion_schedule,
            dino_model=repulsion_dino_model,
        )

        # Repulsion module will be initialized lazily in sample()
        self._repulsion_module = None

        # Pruning configuration (Exp 2)
        # sigma_break is used to detect sigma transition for pruning timing
        self.repulsion_sigma_break = repulsion_sigma_break

        # Pruning debug logs (for pruning.jsonl)
        self.pruning_debug_logs: List[Dict] = []

        # Optimization debug logs (for optimization.jsonl)
        self.optimization_debug_logs: List[Dict] = []

    def _latent_optimization(
        self,
        z: torch.Tensor,
        measurement: torch.Tensor,
        operator: 'LatentWrapper',
        model: nn.Module,
        lr: float = 5e-3,
        eps: float = 1e-3,
        max_iters: int = 500,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        ReSample-style latent space optimization with batch-independent termination.

        Computes argmin_z ||y - A(D(z))||^2 where D is the decoder.

        Key difference from ReSample: Each batch element has independent termination.
        This ensures each sample is optimized independently without being affected
        by other samples in the batch.

        Termination criteria (per-element):
        1. Loss threshold: cur_loss < eps² (1e-6 for eps=1e-3)
        2. Loss plateau: After 200 iterations, if init_loss < cur_loss, stop

        Args:
            z: Latent tensor [B, C, H, W]
            measurement: Measurement tensor [B, ...]
            operator: LatentWrapper containing forward operator
            model: Latent diffusion model with decode()
            lr: Learning rate (default: 5e-3)
            eps: Tolerance (default: 1e-3, squared = 1e-6)
            max_iters: Maximum iterations (default: 500)
            verbose: Print progress

        Returns:
            Tuple of (optimized_z, info_dict)
        """
        batch_size = z.shape[0]
        device = z.device

        # Store original z for accept-if-improve logic
        z_original = z.detach().clone()

        # Clone z and enable gradient
        z_opt = z.detach().clone().requires_grad_(True)

        # Initialize optimizer
        optimizer = torch.optim.AdamW([z_opt], lr=lr)

        # Per-element tracking
        init_losses = torch.zeros(batch_size, device=device)
        final_losses = torch.zeros(batch_size, device=device)
        final_iters = torch.zeros(batch_size, dtype=torch.long, device=device)
        terminated = torch.zeros(batch_size, dtype=torch.bool, device=device)
        termination_reasons = ['max_iters'] * batch_size  # Default reason

        # Loss history for plateau detection (last 200 iterations per element)
        # We only need init_loss for plateau check

        opt_start_time = time.time()

        for itr in range(max_iters):
            # Skip if all terminated
            if terminated.all():
                break

            optimizer.zero_grad()

            # Decode latent to image space
            x_decoded = model.decode(z_opt)

            # Forward through operator: A(D(z))
            # Note: operator is LatentWrapper, operator.op is the actual measurement operator
            y_pred = operator.op(x_decoded)

            # Compute per-element MSE loss
            # Reshape to [B, -1] and compute mean over non-batch dims
            y_pred_flat = y_pred.view(batch_size, -1)
            measurement_flat = measurement.view(batch_size, -1)
            per_element_loss = ((y_pred_flat - measurement_flat) ** 2).mean(dim=1)  # [B]

            # Store initial losses
            if itr == 0:
                init_losses = per_element_loss.detach().clone()

            # Compute total loss for backward (only for non-terminated elements)
            active_mask = ~terminated
            if active_mask.any():
                # Weighted mean: only compute gradient for active elements
                total_loss = (per_element_loss * active_mask.float()).sum() / active_mask.float().sum()
                total_loss.backward()

                # Zero out gradients for terminated elements
                if z_opt.grad is not None:
                    z_opt.grad.data[terminated] = 0

                optimizer.step()

            # Check termination criteria per element
            cur_losses = per_element_loss.detach()

            # Criterion 1: Loss threshold (cur_loss < eps²)
            loss_threshold_met = cur_losses < (eps ** 2)
            newly_terminated_threshold = loss_threshold_met & ~terminated
            if newly_terminated_threshold.any():
                for idx in newly_terminated_threshold.nonzero(as_tuple=True)[0]:
                    termination_reasons[idx.item()] = 'loss_threshold'
                    final_iters[idx] = itr
                    final_losses[idx] = cur_losses[idx]
                terminated = terminated | newly_terminated_threshold

            # Criterion 2: Loss plateau (after 200 iters, if init_loss < cur_loss)
            if itr >= 200:
                plateau_met = (init_losses < cur_losses) & ~terminated
                if plateau_met.any():
                    for idx in plateau_met.nonzero(as_tuple=True)[0]:
                        termination_reasons[idx.item()] = 'loss_plateau'
                        final_iters[idx] = itr
                        final_losses[idx] = cur_losses[idx]
                    terminated = terminated | plateau_met

            # Progress logging
            if verbose and itr % 50 == 0:
                active_count = (~terminated).sum().item()
                mean_loss = cur_losses[~terminated].mean().item() if active_count > 0 else 0
                print(f"[Optimization] iter={itr}, active={active_count}/{batch_size}, "
                      f"mean_loss={mean_loss:.6f}")

        # Set final values for any remaining active elements
        with torch.no_grad():
            x_decoded = model.decode(z_opt)
            y_pred = operator.op(x_decoded)
            y_pred_flat = y_pred.view(batch_size, -1)
            measurement_flat = measurement.view(batch_size, -1)
            final_cur_losses = ((y_pred_flat - measurement_flat) ** 2).mean(dim=1)

        still_active = ~terminated
        final_losses[still_active] = final_cur_losses[still_active]
        final_iters[still_active] = max_iters

        opt_end_time = time.time()

        # ============================================================
        # Accept-if-improve: element별로 최적화 후 loss가 더 낮은 경우에만 채택
        # final_loss < init_loss → 최적화된 z 사용
        # final_loss >= init_loss → 원래 z 유지 (최적화가 오히려 악화시킴)
        # ============================================================
        accept_mask = final_losses < init_losses  # [B] bool tensor

        # Build final z: element-wise selection
        z_final = z_opt.detach().clone()
        reject_mask = ~accept_mask
        if reject_mask.any():
            # Replace rejected elements with original z
            z_final[reject_mask] = z_original[reject_mask]

        # Track accept/reject counts
        num_accepted = accept_mask.sum().item()
        num_rejected = reject_mask.sum().item()

        # Build info dict
        info = {
            'init_losses': init_losses.cpu().tolist(),
            'final_losses': final_losses.cpu().tolist(),
            'final_iters': final_iters.cpu().tolist(),
            'termination_reasons': termination_reasons,
            'total_time_seconds': opt_end_time - opt_start_time,
            'lr': lr,
            'eps': eps,
            'max_iters': max_iters,
            'batch_size': batch_size,
            # Accept-if-improve info
            'accepted_mask': accept_mask.cpu().tolist(),
            'num_accepted': num_accepted,
            'num_rejected': num_rejected,
        }

        if verbose:
            print(f"[Optimization] Done. Time={info['total_time_seconds']:.2f}s, "
                  f"Mean final_loss={np.mean(info['final_losses']):.6f}, "
                  f"Mean iters={np.mean(info['final_iters']):.1f}, "
                  f"Accepted={num_accepted}/{batch_size}")

        return z_final, info

    def sample(self, model, z_start, operator, measurement, evaluator=None, record=False, verbose=False, **kwargs):
        """
        Performs sampling using LatentDAPS in latent space, decoding intermediate results.

        Supports particle repulsion (Exp 1/3) via score-level injection:
        - Repulsion is computed at each step when active
        - Repulsion gradient is injected into the ODE derivative via pfode.set_repulsion()

        Args:
            model (LatentDiffusionModel): Latent diffusion model.
            z_start (torch.Tensor): Initial latent state tensor.
            operator (nn.Module): Measurement operator applied in data space.
            measurement (torch.Tensor): Observed measurement tensor.
            evaluator (Evaluator, optional): Evaluator for monitoring performance.
            record (bool, optional): Whether to record intermediate states and metrics.
            verbose (bool, optional): Enables progress bar and evaluation metrics.
            **kwargs:
                gt (torch.Tensor, optional): Ground truth data for evaluation.

        Returns:
            torch.Tensor: Final sampled data decoded from latent space.
        """
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.trange(self.annealing_scheduler.num_steps - 1) if verbose else range(self.annealing_scheduler.num_steps - 1)
        warpped_operator = LatentWrapper(operator, model)

        # ============================================================
        # Time Logging: timestep별 소요 시간 측정
        # ============================================================
        total_steps = self.annealing_scheduler.num_steps - 1
        per_step_times = []  # 각 step별 소요 시간 (초)
        sampling_start_time = time.time()

        # ============================================================
        # Repulsion Module Initialization (Exp 1/3)
        # ============================================================
        sigma_max = self.annealing_scheduler.sigma_steps[0].item()  # First (largest) sigma
        if self.repulsion_config.scale > 0:
            device = z_start.device
            if self._repulsion_module is None:
                self._repulsion_module = RepulsionModule(self.repulsion_config, device)
            self._repulsion_module.reset_metrics()
            if verbose:
                print(f"[Repulsion] Enabled with scale={self.repulsion_config.scale}, "
                      f"sigma_break={self.repulsion_config.sigma_break}, schedule={self.repulsion_config.schedule}")

        # Repulsion metrics for logging
        repulsion_step_info = []

        # ============================================================
        # Repulsion Debug Logs (for repulsion.jsonl)
        # Sampling rule: step<50: every 5 steps, step>=50: every 25 steps
        # Always include: step in {0,1,2,5,10}
        # ============================================================
        self.repulsion_debug_logs = []
        always_log_steps = {0, 1, 2, 5, 10}

        # ============================================================
        # Pruning State (Exp 2)
        # - did_prune: 정확히 1회만 pruning 수행을 보장하는 플래그
        # - prev_sigma: sigma 전환 감지를 위한 이전 step sigma 저장
        # - pruning 후 measurement도 함께 slice되어야 함
        # - VRAM 측정: segments 기반 구간별 peak 독립 측정
        #   (향후 Exp4 optimization 추가 시에도 유지보수 용이하도록 설계)
        # ============================================================
        did_prune = False
        prev_sigma = None
        self.pruning_debug_logs = []  # Reset for each image
        self.optimization_debug_logs = []  # Reset for each image (Exp 4)
        self.vram_segments = {}  # segments 기반 VRAM 측정 (예: {'pre_pruning': 10150.0, 'post_pruning': 6100.0, 'optimization': 5800.0})

        def should_log_step(step: int) -> bool:
            if step in always_log_steps:
                return True
            if step < 50:
                return step % 5 == 0
            else:
                return step % 25 == 0

        zt = z_start
        for step in pbar:
            step_start_time = time.time()
            sigma = self.annealing_scheduler.sigma_steps[step]
            sigma_val = sigma.item() if hasattr(sigma, 'item') else float(sigma)

            # ============================================================
            # Gradient Toggle: 실험 1~5에서 필요한 gradient 계산 제어
            # ============================================================
            # Repulsion: repulsion_scale > 0이면 적용 (hard_data_consistency와 무관)
            do_repulsion = (self.repulsion_scale > 0)
            # Check if repulsion is active at this sigma level
            repulsion_active = do_repulsion and self._repulsion_module is not None and \
                               self._repulsion_module.is_active(sigma_val, sigma_max)
            # Note: hard_data_consistency optimization은 loop 종료 후 맨 마지막에 수행됨

            # ============================================================
            # [실험 1/3] Compute Repulsion Gradient
            # ============================================================
            repulsion_grad = None
            repulsion_scale = 0.0
            step_repulsion_info = {'step': step, 'sigma': sigma_val, 'repulsion_active': False}

            if repulsion_active:
                # Compute repulsion gradient in DINO feature space
                # This requires gradient computation for backprop from DINO to latent
                with torch.enable_grad():
                    repulsion_grad, rep_info = self._repulsion_module.compute(
                        latents=zt,
                        decode_fn=model.decode,
                        sigma=sigma_val,
                        sigma_max=sigma_max,
                    )
                    repulsion_scale = rep_info.get('repulsion_scale', 0.0)
                    step_repulsion_info.update(rep_info)
                    repulsion_step_info.append(step_repulsion_info)

                    # DEBUG: Log repulsion info at intervals
                    if verbose and step % 10 == 0:
                        print(f"[Repulsion] step={step}, sigma={sigma_val:.2f}, "
                              f"scale={repulsion_scale:.4f}, "
                              f"mean_dist={rep_info.get('mean_pairwise_distance', 0):.4f}, "
                              f"grad_norm={rep_info.get('repulsion_grad_norm', 0):.4f}")

            # 1. reverse diffusion with repulsion injection
            with torch.no_grad():
                diffusion_scheduler = get_diffusion_scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
                pfode = DiffusionPFODE(model, diffusion_scheduler, solver='euler')

                # Set annealing step context for derivative logging
                pfode.begin_annealing_step(step, sigma_val)

                # Inject repulsion into score if active
                if repulsion_grad is not None and repulsion_scale > 0:
                    pfode.set_repulsion(repulsion_grad.detach(), scale=repulsion_scale)

                z0hat = pfode.sample(zt)

                # ============================================================
                # Collect debug info from derivative for repulsion.jsonl
                # ============================================================
                if should_log_step(step):
                    score_info = pfode.get_last_score_info()
                    debug_log_entry = {
                        'step': step,
                        'sigma': sigma_val,
                        # From derivative (pfode)
                        'repulsion_on': score_info.get('repulsion_on', False),
                        'repulsion_scale_used': score_info.get('repulsion_scale_used', 0.0),
                        'score_base_norm': score_info.get('score_base_norm', 0.0),
                        'repulsion_norm': score_info.get('repulsion_norm', 0.0),
                        'scaled_repulsion_norm': score_info.get('scaled_repulsion_norm', 0.0),
                        'ratio_scaled_to_score': score_info.get('ratio_scaled_to_score', 0.0),
                        'repulsion_cleared': score_info.get('repulsion_cleared', True),
                    }
                    # Add info from repulsion module (DINO distances, weights)
                    if repulsion_active and step_repulsion_info:
                        debug_log_entry.update({
                            'mean_pairwise_dino_dist': step_repulsion_info.get('mean_pairwise_distance', 0.0),
                            'weights_mean': step_repulsion_info.get('weights_mean', 0.0),
                            'weights_max': step_repulsion_info.get('weights_max', 0.0),
                            'weights_nonzero_frac': step_repulsion_info.get('weights_nonzero_frac', 0.0),
                            'repulsion_time_sec': step_repulsion_info.get('repulsion_time_seconds', 0.0),
                        })
                    self.repulsion_debug_logs.append(debug_log_entry)

                pfode.clear_repulsion()  # Clear repulsion state after sampling
                pfode.end_annealing_step()
                x0hat = model.decode(z0hat)

            # 2. MCMC update (항상 enable_grad - operator.gradient()가 data fitting gradient 계산)
            with torch.enable_grad():
                z0y = self.mcmc_sampler.sample(zt, model, z0hat, warpped_operator, measurement, sigma, step / self.annealing_scheduler.num_steps)

                # TODO [실험 4]: Optimization 적용 시 latent optimization 수행
                # if do_optimization:
                #     z0y = self._latent_optimization(z0y, measurement, operator, ...)

            with torch.no_grad():
                x0y = model.decode(z0y)

            # 3. forward diffusion
            if step != self.annealing_scheduler.num_steps - 1:
                zt = z0y + torch.randn_like(z0y) * self.annealing_scheduler.sigma_steps[step + 1]
            else:
                zt = z0y
            with torch.no_grad():
                xt = model.decode(zt)

            # ============================================================
            # [실험 2] Pruning 적용
            # 트리거 조건 (둘 중 하나 만족 + did_prune=False):
            #   1. step 기반: step == pruning_step
            #   2. 전환 기반: prev_sigma >= sigma_break and curr_sigma < sigma_break
            # 기준: measurement loss가 가장 작은 2개 particle만 유지 (slicing)
            # ============================================================
            if self.pruning_step >= 0 and not did_prune:
                # 전환 기반 트리거 체크
                sigma_transition_triggered = (
                    prev_sigma is not None and
                    prev_sigma >= self.repulsion_sigma_break and
                    sigma_val < self.repulsion_sigma_break
                )
                # step 기반 트리거 체크
                step_triggered = (step == self.pruning_step)

                if step_triggered or sigma_transition_triggered:
                    # ============================================================
                    # VRAM 측정: Pruning 전 peak 기록 후 reset (segments 기반)
                    # ============================================================
                    if torch.cuda.is_available() and not self.use_tpu:
                        self.vram_segments['pre_pruning'] = torch.cuda.max_memory_allocated() / (1024 * 1024)
                        torch.cuda.reset_peak_memory_stats()

                    # Pruning 수행
                    batch_size_before = zt.shape[0]
                    keep_k = 2  # 4 → 2 pruning

                    if batch_size_before > keep_k:
                        # measurement loss 계산: warpped_operator.loss()는 (B,) shape 반환
                        with torch.no_grad():
                            # z0y를 기준으로 loss 계산 (현재 step의 denoised latent)
                            losses = warpped_operator.loss(z0y, measurement)  # shape: (B,)

                        # loss가 가장 작은 top-k 선택 (largest=False)
                        _, kept_indices = torch.topk(losses, k=keep_k, largest=False)
                        kept_indices = kept_indices.sort().values  # 인덱스 정렬 (일관성)

                        # Slicing: 관련 텐서들 모두 줄이기
                        zt = zt[kept_indices]
                        z0y = z0y[kept_indices]
                        z0hat = z0hat[kept_indices]
                        x0y = x0y[kept_indices]
                        x0hat = x0hat[kept_indices]
                        xt = xt[kept_indices]
                        measurement = measurement[kept_indices]

                        # 탈락한 particle indices 계산 (logging과 trajectory 모두에 사용)
                        kept_indices_cpu = kept_indices.cpu()
                        all_indices = set(range(batch_size_before))
                        pruned_indices_list = sorted(all_indices - set(kept_indices_cpu.tolist()))

                        # Trajectory도 함께 슬라이싱 (이전 step들의 기록도 batch dimension 맞춤)
                        # 이렇게 하지 않으면 compile() 시 torch.stack에서 shape mismatch 발생
                        if record and hasattr(self, 'trajectory'):
                            pruned_indices_cpu = torch.tensor(pruned_indices_list)

                            # 탈락한 particle들의 trajectory를 별도 저장 (pruning 시점까지)
                            self.pruned_trajectory = Trajectory()
                            for key in self.trajectory.tensor_data:
                                self.pruned_trajectory.tensor_data[key] = [
                                    t[pruned_indices_cpu] for t in self.trajectory.tensor_data[key]
                                ]
                            # value_data도 복사 (sigma 등)
                            for key in self.trajectory.value_data:
                                self.pruned_trajectory.value_data[key] = self.trajectory.value_data[key].copy()

                            # 살아남은 particle들의 trajectory만 유지
                            for key in self.trajectory.tensor_data:
                                self.trajectory.tensor_data[key] = [
                                    t[kept_indices_cpu] for t in self.trajectory.tensor_data[key]
                                ]

                        # Logging
                        prune_log_entry = {
                            'prune_step': step,
                            'prev_sigma': prev_sigma if prev_sigma is not None else sigma_val,
                            'curr_sigma': sigma_val,
                            'repulsion_sigma_break': self.repulsion_sigma_break,
                            'losses': losses.cpu().tolist(),
                            'kept_indices': kept_indices.cpu().tolist(),
                            'pruned_indices': pruned_indices_list,
                            'kept_losses': losses[kept_indices].cpu().tolist(),
                            'pruned_losses': [losses[i].item() for i in pruned_indices_list],
                            'batch_before': batch_size_before,
                            'batch_after': zt.shape[0],
                            'did_prune': True,
                            'trigger': 'step' if step_triggered else 'sigma_transition',
                            'mode': 'slicing',
                        }
                        self.pruning_debug_logs.append(prune_log_entry)

                        if verbose:
                            print(f"[Pruning] step={step}, sigma={sigma_val:.4f}, "
                                  f"batch: {batch_size_before} → {zt.shape[0]}, "
                                  f"kept_indices={kept_indices.cpu().tolist()}, "
                                  f"losses={[f'{l:.4f}' for l in losses.cpu().tolist()]}")

                    did_prune = True  # 1회만 수행

            # prev_sigma 업데이트 (다음 step의 전환 감지용)
            prev_sigma = sigma_val

            # 4. evaluation
            x0hat_results = x0y_results = {}
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    # Pruning 후 batch size가 줄었을 경우 gt도 맞춰서 슬라이싱
                    # (gt는 이미지 배치, 처음에 num_samples로 복제됨)
                    current_batch_size = x0hat.shape[0]
                    if gt.shape[0] != current_batch_size:
                        gt_eval = gt[:current_batch_size]
                    else:
                        gt_eval = gt
                    x0hat_results = evaluator(gt_eval, measurement, x0hat)
                    x0y_results = evaluator(gt_eval, measurement, x0y)

                # record
                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        'x0hat' + '_' + main_eval_fn_name: f"{x0hat_results[main_eval_fn_name].item():.2f}",
                        'x0y' + '_' + main_eval_fn_name: f"{x0y_results[main_eval_fn_name].item():.2f}",
                    })
            if record:
                self._record(xt, x0y, x0hat, sigma, x0hat_results, x0y_results)

            # Time Logging: step 종료 시간 기록
            step_end_time = time.time()
            per_step_times.append(step_end_time - step_start_time)

            # TPU: 매 step 끝에서 mark_step() 호출 (lazy execution 그래프 실행)
            mark_step(self.use_tpu)

        # Time Logging: 전체 샘플링 시간 및 통계 저장
        sampling_end_time = time.time()
        self.timing_info = {
            'total_seconds': sampling_end_time - sampling_start_time,
            'total_steps': total_steps,
            'per_step_seconds': per_step_times,
            'mean_step_seconds': np.mean(per_step_times),
            'std_step_seconds': np.std(per_step_times),
        }

        # ============================================================
        # Repulsion Metrics (Exp 1/3)
        # ============================================================
        if self._repulsion_module is not None:
            self.repulsion_info = self._repulsion_module.get_summary_metrics()
            self.repulsion_info['step_details'] = repulsion_step_info
            if verbose and len(repulsion_step_info) > 0:
                print(f"[Repulsion] Summary: {self.repulsion_info['repulsion_active_steps']} active steps, "
                      f"total time: {self.repulsion_info['repulsion_total_time_seconds']:.2f}s")
        else:
            self.repulsion_info = {'repulsion_enabled': False}

        # ============================================================
        # VRAM 측정: Pruning 후 구간 peak 기록 (Exp 2, segments 기반)
        # ============================================================
        if self.pruning_step >= 0 and did_prune:
            if torch.cuda.is_available() and not self.use_tpu:
                self.vram_segments['post_pruning'] = torch.cuda.max_memory_allocated() / (1024 * 1024)
            if verbose and self.vram_segments:
                print(f"[VRAM] segments: {self.vram_segments}")

        # ============================================================
        # [실험 4] Hard Data Consistency Optimization (ReSample-style)
        # 맨 마지막 timestep에서만 latent space optimization 수행
        # - Loss: || A(decode(z)) - y ||^2
        # - Termination: (1) cur_loss < eps², (2) 200 iter 후 init_loss < cur_loss
        # - Batch element 간 independent termination
        # ============================================================
        if self.hard_data_consistency == 1:
            if verbose:
                print(f"[Optimization] Starting ReSample-style latent optimization...")

            # VRAM 측정: Optimization 전 peak 기록 후 reset
            if torch.cuda.is_available() and not self.use_tpu:
                # Note: pruning이 있었으면 post_pruning이 이미 기록됨
                # optimization 구간을 별도로 측정
                torch.cuda.reset_peak_memory_stats()

            # Get optimization hyperparameters from kwargs or use defaults
            opt_lr = kwargs.get('optimization_lr', 5e-3)
            opt_eps = kwargs.get('optimization_eps', 1e-3)
            opt_max_iters = kwargs.get('optimization_max_iters', 500)

            # Perform optimization on the final latent
            # At the last step: zt = z0y (no noise added), so zt is already the final latent
            z_final = zt.detach().clone()

            z_optimized, opt_info = self._latent_optimization(
                z=z_final,
                measurement=measurement,
                operator=warpped_operator,
                model=model,
                lr=opt_lr,
                eps=opt_eps,
                max_iters=opt_max_iters,
                verbose=verbose,
            )

            # Decode optimized latent back to image space
            with torch.no_grad():
                xt = model.decode(z_optimized)

            # Store optimization info for logging
            self.optimization_debug_logs.append(opt_info)
            self.optimization_info = opt_info

            # VRAM 측정: Optimization 후 peak 기록
            if torch.cuda.is_available() and not self.use_tpu:
                self.vram_segments['optimization'] = torch.cuda.max_memory_allocated() / (1024 * 1024)

            if verbose:
                print(f"[Optimization] Complete. Mean iters={np.mean(opt_info['final_iters']):.1f}, "
                      f"Time={opt_info['total_time_seconds']:.2f}s")
                if self.vram_segments:
                    print(f"[VRAM] segments: {self.vram_segments}")
        else:
            self.optimization_info = {'optimization_enabled': False}

        return xt

