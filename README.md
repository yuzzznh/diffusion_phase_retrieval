# Exploring Latent-Space Posterior Sampling Strategies for Diffusion-Based Inverse Problems

**Integrating Hard Data Consistency with Latent HMC**

**Author:** Yujin Kim

---

## Abstract

This repository investigates strategies for improving diffusion-based inverse problem solvers, focusing on the integration of **LatentDAPS** (Decoupled Annealing Posterior Sampling with Latent HMC) and **ReSample** (Hard Data Consistency via Optimization). We explore whether combining these orthogonal approaches—MCMC-based exploration and gradient-based optimization—can yield superior reconstruction quality across various inverse problems.

Our experiments reveal critical insights about the interaction between data consistency optimization and problem-specific characteristics, particularly in **phase retrieval** where measurement information is fundamentally incomplete.

---

## Repository Structure

This repository contains two integrated codebases:

```
diffusion_phase_retrieval/
├── DAPS/                           # Part 2: Phase Retrieval Exploration (Main Focus)
│   ├── posterior_sample.py         # Main inference script
│   ├── sampler.py                  # LatentDAPS + Repulsion + Optimization
│   ├── repulsion.py                # SVGD-style particle repulsion (DINO features)
│   ├── cores/                      # MCMC, schedulers, trajectory
│   ├── commands_gpu/               # Experiment shell scripts (exp0~exp5)
│   ├── results/                    # Experiment outputs
│   └── PROJECT.md                  # Detailed experiment log & analysis
│
└── project_resample_daps/          # Part 1: General Inverse Tasks
    ├── DAPS/                       # LatentDAPS codebase
    ├── resample/                   # ReSample codebase
    └── CLAUDE.md                   # Experiment notes & results
```

---

## Part 1: General Inverse Tasks (Merged Codebase)

### Background

We aimed to combine the strengths of:
- **LatentDAPS**: MCMC-based posterior sampling with noise annealing for global exploration
- **ReSample**: Optimization-based hard data consistency for precise measurement matching

### Methodology

We integrated fixed-iteration gradient descent optimization during the HMC sampling process at timestep interval $t \in [15, 45)$ out of 50 annealing steps.

| Optimization Iterations | Description |
|------------------------|-------------|
| 5 | Light optimization |
| 15 | Moderate optimization |
| 30 | Heavy optimization |

### Results Summary

Experiments were conducted on FFHQ and ImageNet (100 images each) across multiple tasks:

| Task | Method | PSNR | SSIM | LPIPS |
|------|--------|------|------|-------|
| **Phase Retrieval** | LatentDAPS (HMC) | 30.13 | 0.839 | 0.192 |
| **HDR** | LatentDAPS + ReSample(5) | 27.19 | 0.845 | 0.203 |
| **Nonlinear Blur (FFHQ)** | LatentDAPS + ReSample(5) | 30.03 | 0.848 | 0.196 |
| **Nonlinear Blur (ImageNet)** | LatentDAPS + ReSample(5) | 27.13 | 0.750 | 0.252 |

*Source: `project_resample_daps/CLAUDE.md`*

### Key Findings

1. **General Improvement**: Optimization integration improved PSNR/SSIM for most linear tasks (Gaussian blur, Motion blur, Inpainting)
2. **Super Resolution Degradation**: SR showed performance drops with optimization, suggesting task-specific tuning is necessary
3. **LPIPS Stagnation**: Perceptual quality (LPIPS) improvements were minimal across all tasks

---

## Part 2: Exploration on Phase Retrieval (Main Focus)

### Background

Phase retrieval is a fundamentally **multimodal inverse problem** where measurement $y = |Ax|$ contains only amplitude information, losing all phase information. This creates symmetric solutions at 0° and 180° phase shifts that are indistinguishable from the measurement alone.

### Experimental Framework

We designed a systematic 5-experiment framework to investigate multi-particle sampling strategies:

| Experiment | Configuration | Purpose |
|------------|---------------|---------|
| **Exp0** | 4 independent runs (Baseline) | LatentDAPS baseline |
| **Exp1** | 4 particles with Repulsion | SVGD-style diversity enforcement |
| **Exp2** | 4→2 Pruning | Efficiency via particle elimination |
| **Exp3** | 2 particles with Repulsion | Ablation on particle count |
| **Exp4** | Exp1 + Hard Data Consistency | ReSample optimization at final step |

### Repulsion Implementation

We implemented SVGD-style repulsion using DINO-ViT features:
- **Feature Extractor**: `dino_vits16` (same as RLSD)
- **Kernel**: RBF with median heuristic bandwidth
- **Injection Point**: Score-level addition in diffusion ODE
- **Active Region**: $\sigma \in [1.0, 10.0]$ (approximately 60% of sampling steps)

### Results (ImageNet, scale=10)

#### 10 Image Benchmark

| Experiment | Best PSNR | Mean PSNR | Time | VRAM |
|------------|-----------|-----------|------|------|
| **Exp0 Baseline** | 17.50 dB | 15.49 dB | 9,060s | 10,161 MB |
| **Exp1 Repulsion (4p)** | **17.62 dB** | 15.56 dB | 9,191s (+1.4%) | 10,252 MB |
| **Exp3 Repulsion (2p)** | 15.98 dB | 14.71 dB | 4,694s (-48%) | 6,066 MB |
| **Exp4 Optimization** | 17.31 dB | 15.25 dB | +174s overhead | - |

#### 100 Image Final Evaluation (Exp1)

| Metric | 10 Images | 90 Images | Combined 100 |
|--------|-----------|-----------|--------------|
| Best PSNR | 17.62 dB | 15.53 dB | - |

*Source: `DAPS/PROJECT.md`*

### Critical Findings

#### 1. Measurement Loss Optimization Degrades Phase Retrieval

**Observation**: Despite measurement loss decreasing by ~12%, PSNR decreased by up to 8.37 dB for some samples.

| Sample | Init Loss | Final Loss | PSNR Change |
|--------|-----------|------------|-------------|
| 0 | 0.00294 | 0.00257 (-12.6%) | +3.38 dB |
| **1** | 0.00293 | 0.00256 (-12.6%) | **-8.37 dB** |
| 2 | 0.00291 | 0.00255 (-12.4%) | -0.13 dB |
| 3 | 0.00290 | 0.00256 (-11.7%) | +0.45 dB |

**Root Cause**: In phase retrieval, measurement $y = |Ax|$ contains **no phase information**. Optimizing amplitude-only loss can corrupt the correct phase alignment discovered by MCMC sampling.

**Implication**: Accept-if-improve criteria based on measurement loss (`final_loss < init_loss`) does **not guarantee PSNR improvement** for phase retrieval.

#### 2. Repulsion Provides Marginal Improvement

| Scale | Best PSNR | Pairwise Distance | Step 0 Ratio |
|-------|-----------|-------------------|--------------|
| 0.1 | 11.24 dB | 32.13 | ~0.001 |
| **10** | **20.66 dB** | 70.70 | 0.0625 |
| 50 | 6.49 dB | 128.94 | 1.53 (collapsed) |

- Scale=10 found as sweet spot: ratio ≈ 6% of score magnitude
- Pairwise distance increased 2.2x (32 → 71)
- But statistical improvement over 10 images was only **+0.12 dB** vs baseline

#### 3. Independent Runs Outperform Repulsive Runs

Counterintuitively, 4 independent runs (Exp0) performed comparably to 4 repulsive particles (Exp1):
- The additional computational overhead of DINO feature extraction and repulsion gradient computation did not justify the marginal gains
- 2 repulsive particles (Exp3) performed significantly worse than 4 independent runs

---

## Conclusions

1. **Task-Specific Optimization**: Hard data consistency optimization is beneficial for tasks with complete measurement information (inpainting, deblurring) but **harmful for phase retrieval** where phase information is absent.

2. **Exploration vs Exploitation Trade-off**: For multimodal problems like phase retrieval, maintaining **diverse exploration through independent runs** may be more robust than enforcing diversity through explicit repulsion mechanisms.

3. **Efficiency Considerations**: 2-particle configurations save ~50% time and ~40% VRAM but sacrifice ~1.5 dB in Best PSNR, suggesting that particle count matters more than inter-particle interactions.

---

## Data & Results

All experiment data is available at: [Google Drive](https://drive.google.com/drive/folders/1gsKhgytUPhjzFLcqWzY7tCDv6jL1UjNs?usp=sharing)

| File | Size | Description |
|------|------|-------------|
| `every_combined_and_ldm.zip` | 14.7 GB | Part 1: General Tasks (Original Experiments) |
| `hmc_resample_independent_1217.zip` | 1.0 GB | Part 1: General Tasks (Additional Experiments) |
| `diffusionpr1216.zip` | 7.2 GB | Part 2: Phase Retrieval Exploration |

---

## References

- **DAPS**: Zhang et al., "Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing" (CVPR 2025 Oral)
- **ReSample**: Song et al., "Solving Inverse Problems with Latent Diffusion Models via Hard Data Consistency" (ICLR 2024)
- **RLSD**: Zilberstein et al., "Repulsive Latent Score Distillation for Solving Inverse Problems" (arXiv:2406.16683)

---

## Acknowledgments

This project builds upon the official codebases of DAPS and ReSample. We thank the original authors for making their implementations publicly available.
