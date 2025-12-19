import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def is_per_sample_data(data):
    """Check if data is per-sample (list of lists) or batch-level (flat list)."""
    if not data:
        return False
    return isinstance(data[0], list)


def visualize_tracking(data, save_path, first_ten=False):
    """Generate tracking visualization PNG (2x3 layout) with overlaid samples."""
    if 'tracking' not in data:
        print("No tracking data available in metrics.json")
        return False

    tracking = data['tracking']

    # Extract data
    opt_iters_all = tracking.get('opt_iters_per_step', [])
    error_before_all = tracking.get('error_before_opt', [])
    error_after_all = tracking.get('error_after_opt', [])
    dist_to_gt_all = tracking.get('dist_to_gt', [])
    mcmc_drift_all = tracking.get('drift_during_mcmc', [])
    latent_norm_all = tracking.get('latent_norm', [])

    if not opt_iters_all:
        print("No opt_iters_per_step data available")
        return False

    # Validate all data is per-sample format
    if not is_per_sample_data(opt_iters_all):
        print("ERROR: opt_iters_per_step is not per-sample format. Legacy batch-level data detected.")
        print("Please re-run experiments with updated sampler.py for per-sample independent optimization.")
        return False
    if error_before_all and not is_per_sample_data(error_before_all):
        print("ERROR: error_before_opt is not per-sample format. Legacy batch-level data detected.")
        return False
    if error_after_all and not is_per_sample_data(error_after_all):
        print("ERROR: error_after_opt is not per-sample format. Legacy batch-level data detected.")
        return False

    # Get n_steps from per-sample data
    n_steps = len(opt_iters_all[0]) if opt_iters_all else 0
    steps = np.arange(n_steps)

    # Determine N (number of unique images) and num_runs
    num_runs = 1
    N = len(opt_iters_all)
    if 'psnr' in data and 'sample' in data['psnr']:
        psnr_sample = data['psnr']['sample']  # [B, num_runs]
        if psnr_sample and isinstance(psnr_sample[0], list):
            num_runs = len(psnr_sample[0])
            N = len(psnr_sample)

    # For selecting best run per sample if num_runs > 1
    def select_best_run(all_data):
        if not all_data or len(all_data) != N * num_runs:
            return all_data
        if num_runs == 1:
            return all_data
        psnr_sample = np.array(data['psnr']['sample'])  # [B, num_runs]
        best_run_idx = np.argmax(psnr_sample, axis=1)  # [B]
        selected = []
        for img_idx in range(N):
            best_run = best_run_idx[img_idx]
            sample_idx = img_idx * num_runs + best_run
            selected.append(all_data[sample_idx])
        return selected

    # Select best run for all per-sample data
    opt_iters_selected = select_best_run(opt_iters_all)
    error_before_selected = select_best_run(error_before_all)
    error_after_selected = select_best_run(error_after_all)
    dist_to_gt_selected = select_best_run(dist_to_gt_all)
    mcmc_drift_selected = select_best_run(mcmc_drift_all)
    latent_norm_selected = select_best_run(latent_norm_all)

    # Apply first_ten filter (first 10 steps)
    if first_ten:
        n_steps = min(10, n_steps)
        steps = np.arange(n_steps)
        opt_iters_selected = [d[:n_steps] for d in opt_iters_selected] if opt_iters_selected else []
        error_before_selected = [d[:n_steps] for d in error_before_selected] if error_before_selected else []
        error_after_selected = [d[:n_steps] for d in error_after_selected] if error_after_selected else []
        dist_to_gt_selected = [d[:n_steps] for d in dist_to_gt_selected] if dist_to_gt_selected else []
        mcmc_drift_selected = [d[:n_steps] for d in mcmc_drift_selected] if mcmc_drift_selected else []
        latent_norm_selected = [d[:n_steps] for d in latent_norm_selected] if latent_norm_selected else []

    # Find ReSample start step (first step with opt_iters > 0)
    resample_start = None
    if opt_iters_selected:
        for i, iters in enumerate(opt_iters_selected[0]):
            if iters > 0:
                resample_start = i
                break

    # Number of samples for overlay
    n_samples = len(opt_iters_selected) if opt_iters_selected else 1
    alpha = max(0.05, min(1.0, 1.0 / n_samples))

    # Create 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    suffix = " (first 10 steps)" if first_ten else ""
    run_info = f", best of {num_runs} runs" if num_runs > 1 else ""
    fig.suptitle(f"Tracking Analysis ({n_samples} samples{run_info}){suffix}",
                 fontsize=14, fontweight='bold')

    # Helper function to add ReSample start line
    def add_resample_line(ax, label=True):
        if resample_start is not None:
            ax.axvline(x=resample_start, color='red', linestyle='--', linewidth=1.5,
                      label=f'ReSample starts (step {resample_start})' if label else None)

    # 1. ReSample Iterations per Step - overlay N samples
    ax1 = axes[0, 0]
    if opt_iters_selected:
        all_vals = [v for d in opt_iters_selected for v in d if v is not None]
        if all_vals:
            ymax = max(all_vals)
            ax1.set_ylim(0, ymax * 1.1)

        for i, d in enumerate(opt_iters_selected):
            label = 'samples' if i == 0 else None
            ax1.bar(steps, d, color='steelblue', alpha=alpha, edgecolor='black', linewidth=0.3, label=label)
        add_resample_line(ax1)
    ax1.set_xlabel('Annealing Step')
    ax1.set_ylabel('Optimization Iterations')
    ax1.set_title('1. ReSample Iterations per Step')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # 2. Measurement Error (Before/After ReSample) - overlay N samples
    ax2 = axes[0, 1]
    if error_before_selected and error_after_selected:
        all_before = [v for d in error_before_selected for v in d if v is not None]
        all_after = [v for d in error_after_selected for v in d if v is not None]
        all_vals = all_before + all_after
        if all_vals:
            ymin, ymax = min(all_vals), max(all_vals)
            margin = (ymax - ymin) * 0.1
            ax2.set_ylim(ymin - margin, ymax + margin)

        for i, (before, after) in enumerate(zip(error_before_selected, error_after_selected)):
            label_b = 'Before Opt' if i == 0 else None
            label_a = 'After Opt' if i == 0 else None
            ax2.plot(steps, before, 'o-', color='red', alpha=alpha, markersize=3, label=label_b)
            ax2.plot(steps, after, 'o-', color='green', alpha=alpha, markersize=3, label=label_a)
        add_resample_line(ax2, label=False)
    ax2.set_xlabel('Annealing Step')
    ax2.set_ylabel(r'Measurement Error $\|y - A(x)\|^2$')
    ax2.set_title('2. Measurement Error (Before/After)')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(alpha=0.3)

    # 3. Distance to GT - overlay N samples
    ax3 = axes[0, 2]
    if dist_to_gt_selected:
        all_vals = [v for d in dist_to_gt_selected for v in d if v is not None]
        if all_vals:
            ymin, ymax = min(all_vals), max(all_vals)
            margin = (ymax - ymin) * 0.1
            ax3.set_ylim(ymin - margin, ymax + margin)

        for i, d in enumerate(dist_to_gt_selected):
            label = 'samples' if i == 0 else None
            ax3.plot(steps, d, 'o-', color='purple', alpha=alpha, markersize=3, label=label)
        add_resample_line(ax3)
    ax3.set_xlabel('Annealing Step')
    ax3.set_ylabel('MSE to Ground Truth')
    ax3.set_title('3. Distance to GT (z0y vs z_GT)')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(alpha=0.3)

    # 4. MCMC Drift - overlay N samples (log scale)
    ax4 = axes[1, 0]
    if mcmc_drift_selected:
        all_vals = [v for d in mcmc_drift_selected for v in d if v is not None and v > 0]
        if all_vals:
            ymin, ymax = min(all_vals), max(all_vals)
            ax4.set_ylim(ymin * 0.5, ymax * 2)

        for i, d in enumerate(mcmc_drift_selected):
            label = 'samples' if i == 0 else None
            ax4.plot(steps, d, 'o-', color='orange', alpha=alpha, markersize=3, label=label)
        add_resample_line(ax4)
        ax4.set_yscale('log')
    ax4.set_xlabel('Annealing Step')
    ax4.set_ylabel(r'MCMC Drift $\|z0y - z0hat\|^2$')
    ax4.set_title('4. MCMC Drift (How much MCMC moves)')
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(alpha=0.3)

    # 5. Latent Norm - overlay N samples
    ax5 = axes[1, 1]
    if latent_norm_selected:
        all_vals = [v for d in latent_norm_selected for v in d if v is not None]
        if all_vals:
            ymin, ymax = min(all_vals), max(all_vals)
            margin = (ymax - ymin) * 0.1
            ax5.set_ylim(ymin - margin, ymax + margin)

        for i, d in enumerate(latent_norm_selected):
            label = 'samples' if i == 0 else None
            ax5.plot(steps, d, 'o-', color='teal', alpha=alpha, markersize=3, label=label)
        add_resample_line(ax5)
    ax5.set_xlabel('Annealing Step')
    ax5.set_ylabel(r'$\|z0y\|_2$')
    ax5.set_title('5. Latent Norm')
    ax5.legend(loc='best', fontsize=8)
    ax5.grid(alpha=0.3)

    # 6. Correlation: dist_to_gt vs latent_norm - overlay N samples (normalized)
    ax6 = axes[1, 2]
    if dist_to_gt_selected and latent_norm_selected:
        dist_all_vals = [v for d in dist_to_gt_selected for v in d if v is not None]
        lat_all_vals = [v for d in latent_norm_selected for v in d if v is not None]

        if dist_all_vals and lat_all_vals:
            dist_min, dist_max = min(dist_all_vals), max(dist_all_vals)
            lat_min, lat_max = min(lat_all_vals), max(lat_all_vals)

            for i, (d, l) in enumerate(zip(dist_to_gt_selected, latent_norm_selected)):
                d_norm = (np.array(d) - dist_min) / (dist_max - dist_min + 1e-8)
                l_norm = (np.array(l) - lat_min) / (lat_max - lat_min + 1e-8)
                label_d = 'dist_to_gt (norm)' if i == 0 else None
                label_l = 'latent_norm (norm)' if i == 0 else None
                ax6.plot(steps, d_norm, 'o-', color='purple', alpha=alpha, markersize=3, label=label_d)
                ax6.plot(steps, l_norm, 'o-', color='teal', alpha=alpha, markersize=3, label=label_l)
            add_resample_line(ax6, label=False)
    ax6.set_xlabel('Annealing Step')
    ax6.set_ylabel('Normalized Value')
    ax6.set_title('6. Correlation: dist_to_gt vs latent_norm')
    ax6.legend(loc='best', fontsize=8)
    ax6.grid(alpha=0.3)

    # Add info text
    if first_ten:
        fig.text(0.01, 0.01, "* Using first 10 steps only", ha='left', va='bottom',
                fontsize=8, style='italic', color='orange')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved tracking visualization to: {save_path}")
    return True


def visualize_timing(data, save_path, first_ten=False):
    """Generate timing visualization PNG (2x2 layout)."""
    if 'tracking' not in data or 'timing' not in data:
        print("No timing/tracking data available in metrics.json")
        return False

    tracking = data['tracking']
    timing = data['timing']

    # Extract timing data
    opt_time_all = tracking.get('opt_time_per_step', [])  # [sample_idx][step] - per-sample
    diffusion_time_all = tracking.get('diffusion_time_per_step', [])  # [batch_idx][step] - per-batch

    # At least one of opt_time or diffusion_time should exist
    # But we can also work with opt_iters if opt_time is missing
    opt_iters_all = tracking.get('opt_iters_per_step', [])

    if not opt_time_all and not diffusion_time_all and not opt_iters_all:
        print("No timing tracking data available")
        return False

    # Validate per-sample data format for opt_time
    if opt_time_all and not is_per_sample_data(opt_time_all):
        print("ERROR: opt_time_per_step is not per-sample format.")
        return False

    # Get n_steps
    n_steps = len(opt_time_all[0]) if opt_time_all else (len(diffusion_time_all[0]) if diffusion_time_all else 0)
    steps = np.arange(n_steps)

    # Determine N and num_runs for best run selection
    num_runs = 1
    N = len(opt_time_all) if opt_time_all else 1
    if 'psnr' in data and 'sample' in data['psnr']:
        psnr_sample = data['psnr']['sample']
        if psnr_sample and isinstance(psnr_sample[0], list):
            num_runs = len(psnr_sample[0])
            N = len(psnr_sample)

    # Select best run per sample
    def select_best_run(all_data):
        if not all_data or len(all_data) != N * num_runs:
            return all_data
        if num_runs == 1:
            return all_data
        psnr_sample = np.array(data['psnr']['sample'])
        best_run_idx = np.argmax(psnr_sample, axis=1)
        selected = []
        for img_idx in range(N):
            best_run = best_run_idx[img_idx]
            sample_idx = img_idx * num_runs + best_run
            selected.append(all_data[sample_idx])
        return selected

    opt_time_selected = select_best_run(opt_time_all)

    # Apply first_ten filter
    if first_ten:
        n_steps = min(10, n_steps)
        steps = np.arange(n_steps)
        opt_time_selected = [d[:n_steps] for d in opt_time_selected] if opt_time_selected else []
        diffusion_time_all = [d[:n_steps] for d in diffusion_time_all] if diffusion_time_all else []

    # Find ReSample start step
    resample_start = None
    if opt_time_selected:
        for i, t in enumerate(opt_time_selected[0]):
            if t is not None and t > 0:
                resample_start = i
                break

    n_samples = len(opt_time_selected) if opt_time_selected else 1
    alpha = max(0.05, min(1.0, 1.0 / n_samples))

    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    suffix = " (first 10 steps)" if first_ten else ""
    run_info = f", best of {num_runs} runs" if num_runs > 1 else ""
    fig.suptitle(f"Timing Analysis ({n_samples} samples{run_info}){suffix}",
                 fontsize=14, fontweight='bold')

    def add_resample_line(ax, label=True):
        if resample_start is not None:
            ax.axvline(x=resample_start, color='red', linestyle='--', linewidth=1.5,
                      label=f'ReSample starts (step {resample_start})' if label else None)

    # 1. Per-sample Optimization Time per Step - overlay N samples
    ax1 = axes[0, 0]
    if opt_time_selected:
        all_vals = [v for d in opt_time_selected for v in d if v is not None and v > 0]
        if all_vals:
            ymax = max(all_vals)
            ax1.set_ylim(0, ymax * 1.1)

        for i, d in enumerate(opt_time_selected):
            label = 'samples' if i == 0 else None
            ax1.bar(steps, d, color='coral', alpha=alpha, edgecolor='black', linewidth=0.3, label=label)
        add_resample_line(ax1)
    ax1.set_xlabel('Annealing Step')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('1. Per-Sample Optimization Time')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # 2. Per-batch Diffusion Time per Step - overlay batches
    ax2 = axes[0, 1]
    if diffusion_time_all:
        n_batches = len(diffusion_time_all)
        batch_alpha = max(0.1, min(1.0, 1.0 / n_batches))

        all_vals = [v for d in diffusion_time_all for v in d if v is not None]
        if all_vals:
            ymax = max(all_vals)
            ax2.set_ylim(0, ymax * 1.1)

        for i, d in enumerate(diffusion_time_all):
            label = 'batches' if i == 0 else None
            ax2.bar(steps, d, color='steelblue', alpha=batch_alpha, edgecolor='black', linewidth=0.3, label=label)
        add_resample_line(ax2, label=False)
    ax2.set_xlabel('Annealing Step')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('2. Per-Batch Diffusion Time')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    # 3. Total Time Distribution per Sample (sum over steps)
    ax3 = axes[1, 0]
    if opt_time_selected:
        total_opt_per_sample = [sum(d) for d in opt_time_selected]
        x = np.arange(len(total_opt_per_sample))
        ax3.bar(x, total_opt_per_sample, color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)
        mean_val = np.mean(total_opt_per_sample)
        std_val = np.std(total_opt_per_sample)
        ax3.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}s')
        ax3.axhspan(mean_val - std_val, mean_val + std_val, alpha=0.2, color='red', label=f'Std: {std_val:.2f}s')
        if len(total_opt_per_sample) <= 20:
            ax3.set_xticks(x)
        else:
            ax3.set_xticks(np.linspace(0, len(total_opt_per_sample) - 1, min(10, len(total_opt_per_sample))).astype(int))
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Total Time (seconds)')
    ax3.set_title('3. Total Optimization Time per Sample')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(axis='y', alpha=0.3)

    # 4. Stacked bar: Per-sample time breakdown (diffusion bottom, opt top)
    ax4 = axes[1, 1]
    if opt_time_selected and diffusion_time_all:
        # Per-sample total opt time
        total_opt_per_sample = [sum(d) for d in opt_time_selected]
        n_samples_plot = len(total_opt_per_sample)

        # Per-batch total diffusion time -> distribute to samples
        # diffusion_time_all: [batch_idx][step], each batch has batch_size samples
        # We need to figure out batch_size from the data
        n_batches = len(diffusion_time_all)
        batch_size = n_samples_plot // n_batches if n_batches > 0 else n_samples_plot

        # Assign diffusion time to each sample (same for all samples in a batch)
        total_diff_per_sample = []
        for batch_idx, batch_diff in enumerate(diffusion_time_all):
            batch_total_diff = sum(batch_diff)
            for _ in range(batch_size):
                total_diff_per_sample.append(batch_total_diff)
        # Handle remaining samples if any
        while len(total_diff_per_sample) < n_samples_plot:
            total_diff_per_sample.append(total_diff_per_sample[-1] if total_diff_per_sample else 0)
        total_diff_per_sample = total_diff_per_sample[:n_samples_plot]

        x = np.arange(n_samples_plot)

        # Stacked bar: diffusion (bottom) + opt (top)
        ax4.bar(x, total_diff_per_sample, color='steelblue', alpha=0.8, label='Diffusion')
        ax4.bar(x, total_opt_per_sample, bottom=total_diff_per_sample, color='coral', alpha=0.8, label='Optimization')

        # Add batch separator lines
        if n_batches > 1 and batch_size > 0:
            for b in range(1, n_batches):
                sep_x = b * batch_size - 0.5
                ax4.axvline(x=sep_x, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # X-ticks
        if n_samples_plot <= 20:
            ax4.set_xticks(x)
        else:
            ax4.set_xticks(np.linspace(0, n_samples_plot - 1, min(10, n_samples_plot)).astype(int))

        avg_total = timing.get('avg_total_time_sec', 0)
        ax4.set_title(f'4. Per-Sample Time Breakdown (Avg Total: {avg_total:.1f}s)')
        ax4.legend(loc='best', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'No timing data available', ha='center', va='center', fontsize=12)
        ax4.set_title('4. Per-Sample Time Breakdown')
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Total Time (seconds)')
    ax4.grid(axis='y', alpha=0.3)

    if first_ten:
        fig.text(0.01, 0.01, "* Using first 10 steps only", ha='left', va='bottom',
                fontsize=8, style='italic', color='orange')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved timing visualization to: {save_path}")
    return True


def visualize_metrics(data, save_path, first_ten=False):
    """Generate visualization PNG for metrics (psnr, ssim, lpips bar charts)."""
    metrics = ['psnr', 'ssim', 'lpips']
    available_metrics = [m for m in metrics if m in data]

    n_metrics = len(available_metrics)
    if n_metrics == 0:
        return

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    suffix = " (first 10)" if first_ten else ""
    fig.suptitle(f"Metrics Distribution{suffix}", fontsize=14, fontweight='bold')

    for ax, metric in zip(axes, available_metrics):
        values = data[metric]['mean'][:10] if first_ten else data[metric]['mean']
        n_samples = len(values)

        # Bar plot
        x = np.arange(n_samples)
        ax.bar(x, values, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)

        # Mean line
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axhspan(mean_val - std_val, mean_val + std_val, alpha=0.2, color='red', label=f'Std: {std_val:.3f}')

        ax.set_xlabel('Sample Index', fontsize=10)
        ax.set_ylabel(metric.upper(), fontsize=10)
        ax.set_title(f'{metric.upper()}', fontsize=12)
        ax.legend(loc='best', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        # Set x-ticks
        if n_samples <= 20:
            ax.set_xticks(x)
        else:
            ax.set_xticks(np.linspace(0, n_samples - 1, min(10, n_samples)).astype(int))

    # Add timing info if available
    if 'timing' in data:
        timing = data['timing']
        timing_text = []
        if 'avg_total_time_sec' in timing:
            timing_text.append(f"Avg Total: {timing['avg_total_time_sec']:.1f}s")
        if 'avg_opt_time_sec' in timing:
            timing_text.append(f"Avg Opt: {timing['avg_opt_time_sec']:.1f}s ({timing.get('avg_opt_ratio_percent', 0):.1f}%)")
        if 'avg_diffusion_time_sec' in timing:
            timing_text.append(f"Avg Diff: {timing['avg_diffusion_time_sec']:.1f}s ({timing.get('avg_diffusion_ratio_percent', 0):.1f}%)")
        if timing_text:
            fig.text(0.99, 0.01, ' | '.join(timing_text), ha='right', va='bottom',
                    fontsize=8, style='italic', color='gray')

    # Add first_ten indicator
    if first_ten:
        fig.text(0.01, 0.01, "* Using first 10 samples only", ha='left', va='bottom',
                fontsize=8, style='italic', color='orange')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics visualization to: {save_path}")


def parse_metrics(json_path, show_std=False, first_ten=False, no_viz=False):
    with open(json_path) as f:
        data = json.load(f)

    metrics = ['psnr', 'ssim', 'lpips']

    # Header
    suffix = " (first 10)" if first_ten else ""
    print(f"\n{'':>8}", end="")
    for metric in metrics:
        print(f" {metric:>10}", end="")
    print(suffix)
    print("-" * (8 + 11 * len(metrics)))

    # Mean row
    print(f"{'Mean':>8}", end="")
    for metric in metrics:
        if metric in data:
            values = data[metric]['mean'][:10] if first_ten else data[metric]['mean']
            mean_val = np.mean(values)
            print(f" {mean_val:>10.3f}", end="")
    print()

    # Std row (optional)
    if show_std:
        print(f"{'Std':>8}", end="")
        for metric in metrics:
            if metric in data:
                values = data[metric]['mean'][:10] if first_ten else data[metric]['mean']
                std_val = np.std(values)
                print(f" {std_val:>10.3f}", end="")
        print()

    # Generate visualizations (default behavior)
    if not no_viz:
        json_path = Path(json_path)
        viz_suffix = "_first10" if first_ten else ""

        # Metrics bar chart
        viz_path = json_path.parent / f"metrics_viz{viz_suffix}.png"
        visualize_metrics(data, viz_path, first_ten)

        # Tracking plot (2x3 layout)
        tracking_path = json_path.parent / f"tracking_viz{viz_suffix}.png"
        visualize_tracking(data, tracking_path, first_ten)

        # Timing plot (2x2 layout)
        timing_path = json_path.parent / f"timing_viz{viz_suffix}.png"
        visualize_timing(data, timing_path, first_ten)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse metrics.json and compute mean values")
    parser.add_argument("json_path", type=str, help="Path to metrics.json")
    parser.add_argument("--std", action="store_true", help="Show standard deviation")
    parser.add_argument("--ten", action="store_true", help="Use only first 10 samples/steps")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization output")

    args = parser.parse_args()
    parse_metrics(args.json_path, args.std, args.ten, args.no_viz)
