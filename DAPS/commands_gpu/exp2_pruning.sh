#!/bin/bash
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# [실험 2] 4 → 2 Pruning (Efficiency Verification)
# 목표: Pruning으로 Time/Memory 절약하면서 성능 유지
# 확인 지표: Max PSNR 유지 여부, Time/Memory 단축량
#
# 사용법: bash exp2_pruning.sh [--1] [--10] [--100]
#   --1   : 1 image sanity check (pruning_step 튜닝용)
#   --10  : 10 images main experiment
#   --100 : 100 images final eval
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

set -e

# 인자 파싱
RUN_1=false
RUN_10=false
RUN_100=false

for arg in "$@"; do
    case $arg in
        --1) RUN_1=true ;;
        --10) RUN_10=true ;;
        --100) RUN_100=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

if [ "$RUN_1" = false ] && [ "$RUN_10" = false ] && [ "$RUN_100" = false ]; then
    echo "사용법: bash exp2_pruning.sh [--1] [--10] [--100]"
    echo "  --1   : 1 image sanity check"
    echo "  --10  : 10 images main experiment"
    echo "  --100 : 100 images final eval"
    exit 0
fi

# ============================================================
# [실험 2] Sanity Check - 1 image
# TODO: pruning_step 튜닝 후 값 변경 (현재 25 = 총 50 step 중 절반)
# ============================================================
if [ "$RUN_1" = true ]; then
    echo "========== [실험 2] 1 image sanity check =========="
    python posterior_sample.py \
    +data=test-imagenet \
    +model=imagenet256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/exp2_pruning/imagenet_1img \
    num_samples=4 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=0.1 \
    pruning_step=25 \
    optimization_step=-1 \
    data.end_id=1 \
    name=exp2_sanity_check \
    gpu=0
fi

# ============================================================
# [실험 2] Main Experiment - 10 images
# ============================================================
if [ "$RUN_10" = true ]; then
    echo "========== [실험 2] 10 images main experiment =========="
    python posterior_sample.py \
    +data=test-imagenet \
    +model=imagenet256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/exp2_pruning/imagenet_10img \
    num_samples=4 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=0.1 \
    pruning_step=25 \
    optimization_step=-1 \
    data.end_id=10 \
    name=exp2_10img \
    gpu=0
fi

# ============================================================
# [실험 2] Final Eval - 100 images
# ============================================================
if [ "$RUN_100" = true ]; then
    echo "========== [실험 2] 100 images final eval =========="
    python posterior_sample.py \
    +data=test-imagenet \
    +model=imagenet256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/exp2_pruning/imagenet_100img \
    num_samples=4 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=0.1 \
    pruning_step=25 \
    optimization_step=-1 \
    data.end_id=100 \
    name=exp2_100img \
    gpu=0
fi

echo "완료!"
