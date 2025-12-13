import time
import tqdm
import torch
import numpy as np
import torch.nn as nn
from cores.trajectory import Trajectory
from cores.scheduler import get_diffusion_scheduler, DiffusionPFODE
from cores.mcmc import MCMCSampler
from forward_operator import LatentWrapper
from utils import mark_step


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

    Implements posterior sampling using a latent diffusion model combined with MCMC updates
    """
    def sample(self, model, z_start, operator, measurement, evaluator=None, record=False, verbose=False, **kwargs):
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

        zt = z_start
        for step in pbar:
            step_start_time = time.time()
            sigma = self.annealing_scheduler.sigma_steps[step]

            # ============================================================
            # Gradient Toggle: 실험 1~5에서 필요한 gradient 계산 제어
            # ============================================================
            # - MCMC sampler 내부의 operator.gradient()는 항상 gradient 계산 필요
            # - Repulsion/Optimization은 별도 flag로 제어
            # ============================================================
            # Repulsion: optimization_step 이전까지만 적용 (초반 탐색)
            do_repulsion = (self.repulsion_scale > 0) and (self.optimization_step < 0 or step < self.optimization_step)
            # Optimization: optimization_step 이후부터 적용 (후반 정밀화)
            do_optimization = (self.optimization_step >= 0) and (step >= self.optimization_step)

            # 1. reverse diffusion (항상 no_grad - 메모리 절약)
            with torch.no_grad():
                diffusion_scheduler = get_diffusion_scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
                sampler = DiffusionPFODE(model, diffusion_scheduler, solver='euler')
                z0hat = sampler.sample(zt)
                x0hat = model.decode(z0hat)

            # 2. MCMC update (항상 enable_grad - operator.gradient()가 data fitting gradient 계산)
            with torch.enable_grad():
                # TODO [실험 1]: Repulsion 적용 시 repulsive force 추가
                # if do_repulsion:
                #     repulsion_grad = self._compute_repulsion(zt, ...)
                #     zt = zt - self.repulsion_scale * repulsion_grad

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

            # TODO [실험 2]: Pruning 적용
            # if (self.pruning_step >= 0) and (step == self.pruning_step):
            #     zt, measurement = self._prune_particles(zt, measurement, ...)

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

        return xt

