#!/bin/bash
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Overnight Experiment: Exp1 & Exp3 with repulsion_scale grid search
# 총 예상 시간: ~8시간
#
# | 실험   | num_samples | repulsion_scale | 예상 시간 |
# |--------|-------------|-----------------|-----------|
# | Exp1-A | 4           | 0.5             | ~2.5h     |
# | Exp1-B | 4           | 1.0             | ~2.5h     |
# | Exp3-A | 2           | 0.5             | ~1.5h     |
# | Exp3-B | 2           | 1.0             | ~1.5h     |
#
# 사용법: bash overnight_exp1_exp3.sh
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

set -e

echo "======================================================================"
echo "Overnight Experiment Started at $(date)"
echo "======================================================================"

# 공통 설정
SIGMA_BREAK=1.0
SCHEDULE="linear"

# ============================================================
# [Exp1-A] 4 particle, repulsion_scale=0.5, 10 images
# ============================================================
echo ""
echo "======================================================================"
echo "[1/4] Exp1-A: 4 particle, scale=0.5 - Started at $(date)"
echo "======================================================================"
python posterior_sample.py \
    +data=test-imagenet \
    +model=imagenet256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/overnight_1213/exp1_scale0.5 \
    num_samples=4 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=0.5 \
    repulsion_sigma_break=${SIGMA_BREAK} \
    repulsion_schedule=${SCHEDULE} \
    pruning_step=-1 \
    optimization_step=-1 \
    data.end_id=10 \
    name=exp1_scale0.5_10img \
    gpu=0

echo "[1/4] Exp1-A completed at $(date)"

# ============================================================
# [Exp1-B] 4 particle, repulsion_scale=1.0, 10 images
# ============================================================
echo ""
echo "======================================================================"
echo "[2/4] Exp1-B: 4 particle, scale=1.0 - Started at $(date)"
echo "======================================================================"
python posterior_sample.py \
    +data=test-imagenet \
    +model=imagenet256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/overnight_1213/exp1_scale1.0 \
    num_samples=4 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=1.0 \
    repulsion_sigma_break=${SIGMA_BREAK} \
    repulsion_schedule=${SCHEDULE} \
    pruning_step=-1 \
    optimization_step=-1 \
    data.end_id=10 \
    name=exp1_scale1.0_10img \
    gpu=0

echo "[2/4] Exp1-B completed at $(date)"

# ============================================================
# [Exp3-A] 2 particle, repulsion_scale=0.5, 10 images
# ============================================================
echo ""
echo "======================================================================"
echo "[3/4] Exp3-A: 2 particle, scale=0.5 - Started at $(date)"
echo "======================================================================"
python posterior_sample.py \
    +data=test-imagenet \
    +model=imagenet256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/overnight_1213/exp3_scale0.5 \
    num_samples=2 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=0.5 \
    repulsion_sigma_break=${SIGMA_BREAK} \
    repulsion_schedule=${SCHEDULE} \
    pruning_step=-1 \
    optimization_step=-1 \
    data.end_id=10 \
    name=exp3_scale0.5_10img \
    gpu=0

echo "[3/4] Exp3-A completed at $(date)"

# ============================================================
# [Exp3-B] 2 particle, repulsion_scale=1.0, 10 images
# ============================================================
echo ""
echo "======================================================================"
echo "[4/4] Exp3-B: 2 particle, scale=1.0 - Started at $(date)"
echo "======================================================================"
python posterior_sample.py \
    +data=test-imagenet \
    +model=imagenet256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/overnight_1213/exp3_scale1.0 \
    num_samples=2 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=1.0 \
    repulsion_sigma_break=${SIGMA_BREAK} \
    repulsion_schedule=${SCHEDULE} \
    pruning_step=-1 \
    optimization_step=-1 \
    data.end_id=10 \
    name=exp3_scale1.0_10img \
    gpu=0

echo "[4/4] Exp3-B completed at $(date)"

# ============================================================
# 완료
# ============================================================
echo ""
echo "======================================================================"
echo "All experiments completed at $(date)"
echo "======================================================================"
echo ""
echo "Results saved in:"
echo "  - results/overnight_1213/exp1_scale0.5/"
echo "  - results/overnight_1213/exp1_scale1.0/"
echo "  - results/overnight_1213/exp3_scale0.5/"
echo "  - results/overnight_1213/exp3_scale1.0/"
