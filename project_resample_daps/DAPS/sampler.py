import time
import tqdm
import torch
import numpy as np
import torch.nn as nn
from cores.trajectory import Trajectory
from cores.scheduler import get_diffusion_scheduler, DiffusionPFODE
from cores.mcmc import MCMCSampler
from forward_operator import LatentWrapper


# ==============================================================================
# [ReSample] Hard Data Consistency Optimization
# Based on original ReSample latent_optimization logic
# ==============================================================================
def _optimize_single_sample(
    z_single,        # Single latent vector [1, C, H, W]
    y_single,        # Single measurement [1, ...]
    operator,        # 측정 연산자 A
    decoder,         # LDM의 Decoder
    eps=1e-3,        # Threshold for early stopping (stops when loss < eps²)
    max_iters=30,    # Fixed 30 iterations (no early stopping)
    lr=5e-3,         # 학습률 (original ReSample default)
    weight_decay=0.01  # AdamW default weight decay
):
    """
    Optimize a single sample independently.

    Fixed 30 iterations, no early termination.

    Returns:
        z_opt: Optimized latent vector [1, C, H, W]
        num_iters: Number of iterations performed
        error_before: Measurement error before optimization (scalar)
        error_after: Measurement error after optimization (scalar)
        opt_time: Optimization time in seconds (scalar)
    """
    _opt_start = time.time()

    # Calculate error before optimization
    with torch.no_grad():
        x_before = decoder(z_single)
        y_before = operator(x_before)
        error_before = ((y_single - y_before) ** 2).sum().item()

    z_opt = z_single.clone().detach().requires_grad_(True)
    optimizer = torch.optim.AdamW([z_opt], lr=lr, weight_decay=weight_decay)

    # window_size = 200
    # losses_history = []
    num_iters = 0

    with torch.enable_grad():
        for itr in range(max_iters):
            optimizer.zero_grad()

            # 1. Decode & Forward
            x_pixel = decoder(z_opt)
            y_hat = operator(x_pixel)

            # 2. Compute loss
            loss = ((y_single - y_hat) ** 2).sum()
            cur_loss = loss.item()
            num_iters = itr + 1

            # # Termination condition 1: Threshold reached
            # if cur_loss < eps ** 2:
            #     break

            # # Termination condition 2: Sliding window plateau detection
            # losses_history.append(cur_loss)
            # if len(losses_history) > window_size:
            #     old_loss = losses_history[0]
            #     if old_loss < cur_loss:  # Plateau detected (loss increased)
            #         break
            #     losses_history.pop(0)  # Slide window

            # 3. Backprop & Update
            loss.backward()
            optimizer.step()

    # Calculate error after optimization
    with torch.no_grad():
        x_after = decoder(z_opt)
        y_after = operator(x_after)
        error_after = ((y_single - y_after) ** 2).sum().item()

    opt_time = time.time() - _opt_start
    return z_opt.detach(), num_iters, error_before, error_after, opt_time


def perform_resample_optimization(
    z_hat,           # 현재 예측된 Latent Vector (z0_hat) [batch_size, C, H, W]
    measurement_y,   # 실제 관측 데이터 (Ground Truth y) [batch_size, ...]
    operator,        # 측정 연산자 A
    decoder,         # LDM의 Decoder
    eps=1e-3,        # Threshold for early stopping (stops when loss < eps²) --- ReSample's default value
    max_iters=30,    # Fixed 30 iterations (no early stopping)
    lr=5e-3,         # 학습률 (original ReSample default)
    weight_decay=0.01  # AdamW default weight decay
):
    """
    [ReSample Logic] Latent z를 최적화하여 측정값 y와의 오차를 줄임.
    Range: 15 <= step <= 44 (Middle steps, skip last 5)

    SEQUENTIAL per-sample optimization for COMPLETE INDEPENDENCE:
    - Each sample is optimized in a separate for loop iteration
    - No shared optimizer, no shared forward pass, no cross-sample influence
    - Each sample has its own optimizer instance and convergence tracking

    Termination conditions per sample (from original ReSample):
    1. Loss plateau detection via sliding window (window_size=200)
       - After 200 iterations, compare current loss with loss from 200 steps ago
       - If current loss >= old loss, convergence failed -> stop
    2. Threshold reached: loss < eps²

    Returns:
        z_opt: Optimized latent vector [batch_size, C, H, W]
        num_iters_per_sample: Number of iterations performed per sample (list)
        error_before: Measurement error before optimization (per-sample list)
        error_after: Measurement error after optimization (per-sample list)
        opt_time_per_sample: Optimization time per sample in seconds (list)
    """
    batch_size = z_hat.shape[0]

    # Results storage
    z_opt_list = []
    num_iters_per_sample = []
    error_before_per_sample = []
    error_after_per_sample = []
    opt_time_per_sample = []

    # Process each sample SEQUENTIALLY for complete independence
    for i in range(batch_size):
        z_single = z_hat[i:i+1]  # [1, C, H, W]
        y_single = measurement_y[i:i+1]  # [1, ...]

        z_opt_i, num_iters_i, error_before_i, error_after_i, opt_time_i = _optimize_single_sample(
            z_single=z_single,
            y_single=y_single,
            operator=operator,
            decoder=decoder,
            eps=eps,
            max_iters=max_iters,
            lr=lr,
            weight_decay=weight_decay
        )

        z_opt_list.append(z_opt_i)
        num_iters_per_sample.append(num_iters_i)
        error_before_per_sample.append(error_before_i)
        error_after_per_sample.append(error_after_i)
        opt_time_per_sample.append(opt_time_i)

    # Concatenate results back to batch
    z_opt = torch.cat(z_opt_list, dim=0)  # [batch_size, C, H, W]

    return z_opt, num_iters_per_sample, error_before_per_sample, error_after_per_sample, opt_time_per_sample


def perform_resample_optimization_batched(
    z_hat,           # [batch_size, C, H, W]
    measurement_y,   # [batch_size, ...]
    operator,
    decoder,
    eps=1e-3,
    max_iters=500,
    lr=5e-3,
    weight_decay=0.01
):
    """
    [ReSample Logic] Batch-wise latent optimization for faster GPU utilization.

    All samples are processed in parallel with per-sample convergence tracking.
    Converged samples are masked out to avoid unnecessary computation.

    Termination conditions per sample (from original ReSample):
    1. Loss plateau detection via sliding window (window_size=200)
    2. Threshold reached: loss < eps²

    Returns:
        z_opt: Optimized latent vector [batch_size, C, H, W]
        num_iters_per_sample: Number of iterations performed per sample (list)
        error_before: Measurement error before optimization (per-sample list)
        error_after: Measurement error after optimization (per-sample list)
        total_opt_time: Total optimization time in seconds (scalar)
    """
    _opt_start = time.time()
    batch_size = z_hat.shape[0]
    device = z_hat.device

    # Calculate error before optimization
    with torch.no_grad():
        x_before = decoder(z_hat)
        y_before = operator(x_before)
        error_before = ((measurement_y - y_before) ** 2).flatten(1).sum(dim=1)  # [batch_size]
        error_before_list = error_before.tolist()

    # Initialize optimization
    z_opt = z_hat.clone().detach().requires_grad_(True)
    optimizer = torch.optim.AdamW([z_opt], lr=lr, weight_decay=weight_decay)

    # Per-sample tracking
    active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    num_iters = torch.zeros(batch_size, dtype=torch.long, device=device)
    losses_history = [[] for _ in range(batch_size)]  # Per-sample loss history
    window_size = 200

    with torch.enable_grad():
        for itr in range(max_iters):
            # Early exit if all samples converged
            if not active_mask.any():
                break

            optimizer.zero_grad()

            # 1. Decode & Forward (batch-wise)
            x_pixel = decoder(z_opt)
            y_hat = operator(x_pixel)

            # 2. Per-sample loss [batch_size]
            per_sample_loss = ((measurement_y - y_hat) ** 2).flatten(1).sum(dim=1)

            # 3. Check termination conditions & update mask
            # [DISABLED] Early termination disabled - fixed 30 iterations
            # with torch.no_grad():
            #     for i in range(batch_size):
            #         if not active_mask[i]:
            #             continue
            #
            #         cur_loss = per_sample_loss[i].item()
            #         num_iters[i] = itr + 1
            #
            #         # Condition 1: Threshold reached
            #         if cur_loss < eps ** 2:
            #             active_mask[i] = False
            #             continue
            #
            #         # Condition 2: Sliding window plateau detection
            #         losses_history[i].append(cur_loss)
            #         if len(losses_history[i]) > window_size:
            #             old_loss = losses_history[i][0]
            #             if old_loss < cur_loss:  # Plateau detected
            #                 active_mask[i] = False
            #                 continue
            #             losses_history[i].pop(0)
            #
            # # Skip backward if no active samples
            # if not active_mask.any():
            #     break

            # 4. Backward with masked loss (only active samples contribute)
            masked_loss = (per_sample_loss * active_mask.float()).sum()
            masked_loss.backward()

            # 5. Zero out gradients for inactive samples
            with torch.no_grad():
                z_opt.grad[~active_mask] = 0

            optimizer.step()

    # Calculate error after optimization
    with torch.no_grad():
        x_after = decoder(z_opt)
        y_after = operator(x_after)
        error_after = ((measurement_y - y_after) ** 2).flatten(1).sum(dim=1)  # [batch_size]
        error_after_list = error_after.tolist()

    total_opt_time = time.time() - _opt_start

    return z_opt.detach(), num_iters.tolist(), error_before_list, error_after_list, total_opt_time


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

    def __init__(self, annealing_scheduler_config, diffusion_scheduler_config, mcmc_sampler_config):
        """
        Initializes the DAPS sampler with the provided scheduler and sampler configurations.

        Args:
            annealing_scheduler_config (dict): Configuration for annealing scheduler.
            diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
            mcmc_sampler_config (dict): Configuration for MCMC sampler.
        """
        super().__init__()
        annealing_scheduler_config, diffusion_scheduler_config = self._check(annealing_scheduler_config,
                                                                             diffusion_scheduler_config)
        self.annealing_scheduler = get_diffusion_scheduler(**annealing_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.mcmc_sampler = MCMCSampler(**mcmc_sampler_config)

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

    Implements posterior sampling using a latent diffusion model combined with MCMC updates
    """
    def sample(self, model, z_start, operator, measurement, evaluator=None, record=False, verbose=False, resample=False, **kwargs):
        """
        Performs sampling using LatentDAPS in latent space, decoding intermediate results.

        Args:
            model (LatentDiffusionModel): Latent diffusion model.
            z_start (torch.Tensor): Initial latent state tensor.
            operator (nn.Module): Measurement operator applied in data space.
            measurement (torch.Tensor): Observed measurement tensor.
            evaluator (Evaluator, optional): Evaluator for monitoring performance.
            record (bool, optional): Whether to record intermediate states and metrics.
            verbose (bool, optional): Enables progress bar and evaluation metrics.
            resample (bool, optional): Enables ReSample hard data consistency (default: False).
            **kwargs:
                gt (torch.Tensor, optional): Ground truth data for evaluation.

        Returns:
            torch.Tensor: Final sampled data decoded from latent space.
        """
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.trange(self.annealing_scheduler.num_steps - 1) if verbose else range(self.annealing_scheduler.num_steps - 1)
        warpped_operator = LatentWrapper(operator, model)

        # [Timing] Initialize timing stats
        _total_start_time = time.time()
        _total_opt_time = 0.0
        _diffusion_time_per_step = []   # [step] - per-batch diffusion time

        # [Memory] Initialize peak memory tracking (MB)
        _diffusion_peak_memory = 0.0    # Peak memory during diffusion phase
        _optimization_peak_memory = 0.0  # Peak memory during optimization phase

        # [Tracking] Initialize per-step, per-sample tracking lists
        # Structure: [step][sample_idx] -> will be transposed to [sample_idx][step] at the end
        num_steps = self.annealing_scheduler.num_steps
        batch_size = z_start.shape[0]
        _opt_iters_per_step = []        # [step][sample_idx] - per-sample opt iterations
        _opt_time_per_step = []         # [step][sample_idx] - per-sample opt time in seconds
        _error_before_opt = []          # [step][sample_idx] - per-sample error before opt
        _error_after_opt = []           # [step][sample_idx] - per-sample error after opt
        _dist_to_gt = []                # [step][sample_idx]
        _drift_during_mcmc = []         # [step][sample_idx]
        _latent_norm = []               # [step][sample_idx]

        # [Tracking] Compute z_GT once before sampling starts
        z_gt = None
        if 'gt' in kwargs:
            with torch.no_grad():
                z_gt = model.encode(kwargs['gt'])

        zt = z_start
        for step in pbar:
            sigma = self.annealing_scheduler.sigma_steps[step]
            # 1. reverse diffusion
            # [Memory] Reset peak memory before diffusion
            torch.cuda.reset_peak_memory_stats()
            _diffusion_start = time.time()
            with torch.no_grad():
                diffusion_scheduler = get_diffusion_scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
                sampler = DiffusionPFODE(model, diffusion_scheduler, solver='euler')
                z0hat = sampler.sample(zt)
            _diffusion_time_per_step.append(time.time() - _diffusion_start)
            # [Memory] Record peak memory after diffusion (track max across all steps)
            _step_diffusion_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            _diffusion_peak_memory = max(_diffusion_peak_memory, _step_diffusion_peak)

            # ==================================================================
            # [ReSample Integration] Hard Data Consistency
            # Strategy: Skip first 15 steps (High Noise) & last 5 steps (Clean)
            # Range: 15 <= step <= 44, only when resample=True
            # ==================================================================
            if resample and 15 <= step <= 44:
                # 1. Latent Optimization with ReSample logic (BATCH-WISE)
                # (sliding window plateau detection + eps² threshold)
                # [Memory] Reset peak memory before optimization
                torch.cuda.reset_peak_memory_stats()
                z0hat_before_opt = z0hat.clone()  # Save for drift calculation
                z0hat, opt_iters, error_before, error_after, opt_time = perform_resample_optimization_batched(
                    z_hat=z0hat,
                    measurement_y=measurement,
                    operator=operator,
                    decoder=model.decode,
                    eps=1e-3,
                    max_iters=30,
                    lr=0.05
                )

                # [Tracking] Record optimization stats (per-sample)
                _opt_iters_per_step.append(opt_iters)
                # Batched version returns total_opt_time (scalar), distribute evenly for compatibility
                _opt_time_per_step.append([opt_time / batch_size] * batch_size)
                _error_before_opt.append(error_before)
                _error_after_opt.append(error_after)

                # [Timing] Accumulate total optimization time
                _total_opt_time += opt_time

                # [Memory] Record peak memory after optimization (track max across all steps)
                _step_opt_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                _optimization_peak_memory = max(_optimization_peak_memory, _step_opt_peak)

                # 2. Update x0hat (Important!)
                with torch.no_grad():
                    x0hat = model.decode(z0hat)

            else:
                # 기존 LatentDAPS 동작 (resample=False 또는 범위 밖)
                _opt_iters_per_step.append([0] * batch_size)
                _opt_time_per_step.append([0.0] * batch_size)
                _error_before_opt.append([None] * batch_size)
                _error_after_opt.append([None] * batch_size)

                with torch.no_grad():
                    x0hat = model.decode(z0hat)
            # ==================================================================

            # 2. MCMC update
            z0y = self.mcmc_sampler.sample(zt, model, z0hat, warpped_operator, measurement, sigma, step / self.annealing_scheduler.num_steps)

            # [Tracking] Calculate drift_during_mcmc per sample: ||z0y - z0hat||² for each sample
            with torch.no_grad():
                # Shape: [batch_size] -> list of floats
                drift_per_sample = ((z0y - z0hat) ** 2).flatten(1).mean(dim=1).tolist()
                _drift_during_mcmc.append(drift_per_sample)

            # [Tracking] Calculate dist_to_gt per sample: ||z0y - z_gt||² for each sample
            if z_gt is not None:
                with torch.no_grad():
                    # Shape: [batch_size] -> list of floats
                    dist_per_sample = ((z0y - z_gt) ** 2).flatten(1).mean(dim=1).tolist()
                    _dist_to_gt.append(dist_per_sample)
            else:
                _dist_to_gt.append([None] * z0y.shape[0])

            # [Tracking] Calculate latent_norm per sample: ||z0y||₂ for each sample
            with torch.no_grad():
                # Shape: [batch_size] -> list of floats
                latent_norm_per_sample = z0y.norm(p=2, dim=(1, 2, 3)).tolist()
                _latent_norm.append(latent_norm_per_sample)

            with torch.no_grad():
                x0y = model.decode(z0y)

            # 3. forward diffusion
            if step != self.annealing_scheduler.num_steps - 1:
                zt = z0y + torch.randn_like(z0y) * self.annealing_scheduler.sigma_steps[step + 1]
            else:
                zt = z0y
            with torch.no_grad():
                xt = model.decode(zt)

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

        # [Timing] Store timing stats as instance attributes
        _total_time = time.time() - _total_start_time

        # Transpose [step][sample_idx] -> [sample_idx][step] for per-sample tracking
        def transpose_tracking(data):
            """Transpose list of lists from [step][sample] to [sample][step]."""
            if not data or not data[0]:
                return data
            num_samples = len(data[0])
            return [[data[step][sample] for step in range(len(data))] for sample in range(num_samples)]

        # Calculate total opt iters (sum across all samples and steps)
        total_opt_iters = sum(sum(step_iters) for step_iters in _opt_iters_per_step)
        total_diffusion_time = sum(_diffusion_time_per_step)

        self.timing_stats = {
            'total_time': _total_time,
            'opt_time': _total_opt_time,
            'opt_ratio': _total_opt_time / _total_time if _total_time > 0 else 0.0,
            'diffusion_time': total_diffusion_time,
            'diffusion_ratio': total_diffusion_time / _total_time if _total_time > 0 else 0.0,
            'diffusion_time_per_step': _diffusion_time_per_step,  # [step] - per-batch
            'total_opt_iters': total_opt_iters,
            'opt_iters_per_step': transpose_tracking(_opt_iters_per_step),   # [sample_idx][step]
            'opt_time_per_step': transpose_tracking(_opt_time_per_step),     # [sample_idx][step] - per-sample opt time
            'error_before_opt': transpose_tracking(_error_before_opt),   # [sample_idx][step]
            'error_after_opt': transpose_tracking(_error_after_opt),     # [sample_idx][step]
            'dist_to_gt': transpose_tracking(_dist_to_gt),       # [sample_idx][step]
            'drift_during_mcmc': transpose_tracking(_drift_during_mcmc),  # [sample_idx][step]
            'latent_norm': transpose_tracking(_latent_norm),     # [sample_idx][step]
            # [Memory] Peak memory per phase (MB)
            'diffusion_peak_memory_mb': round(_diffusion_peak_memory, 2),
            'optimization_peak_memory_mb': round(_optimization_peak_memory, 2),
        }

        return xt
