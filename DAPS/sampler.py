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
                 repulsion_scale=0.0, pruning_step=-1, optimization_step=-1, use_tpu=False):
        """
        Initializes the DAPS sampler with the provided scheduler and sampler configurations.

        Args:
            annealing_scheduler_config (dict): Configuration for annealing scheduler.
            diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
            mcmc_sampler_config (dict): Configuration for MCMC sampler.
            repulsion_scale (float): Initial strength of particle repulsion. 0.0 = independent (DAPS baseline).
            pruning_step (int): Timestep to perform pruning. -1 = no pruning.
            optimization_step (int): Timestep to start latent optimization. -1 = no optimization.
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
        self.optimization_step = optimization_step

        # TPU/CUDA Flag
        self.use_tpu = use_tpu

        # Gradient 필요 여부 (실험 0에서는 False)
        self.needs_grad = (repulsion_scale > 0) or (optimization_step >= 0)

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
                 pruning_step=-1, optimization_step=-1, use_tpu=False):
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
            optimization_step: Timestep to start optimization (-1 = disabled)
            use_tpu: Whether to use TPU
        """
        super().__init__(
            annealing_scheduler_config=annealing_scheduler_config,
            diffusion_scheduler_config=diffusion_scheduler_config,
            mcmc_sampler_config=mcmc_sampler_config,
            repulsion_scale=repulsion_scale,
            pruning_step=pruning_step,
            optimization_step=optimization_step,
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
        self.vram_segments = {}  # segments 기반 VRAM 측정 (예: {'pre_pruning': 10150.0, 'post_pruning': 6100.0})

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
            # Repulsion: optimization_step 이전까지만 적용 (초반 탐색)
            do_repulsion = (self.repulsion_scale > 0) and (self.optimization_step < 0 or step < self.optimization_step)
            # Check if repulsion is active at this sigma level
            repulsion_active = do_repulsion and self._repulsion_module is not None and \
                               self._repulsion_module.is_active(sigma_val, sigma_max)
            # Optimization: optimization_step 이후부터 적용 (후반 정밀화)
            do_optimization = (self.optimization_step >= 0) and (step >= self.optimization_step)

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

        return xt

