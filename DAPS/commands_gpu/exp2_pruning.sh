#!/bin/bash
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# [실험 2] 4 → 2 Pruning (Efficiency Verification)
# 목표: Pruning으로 Time/Memory 절약하면서 성능 유지
# 확인 지표: Max PSNR 유지 여부, Time/Memory 단축량
#
# 사용법: bash exp2_pruning.sh [--1] [--10] [--90]
#   --1   : 1 image sanity check (이미지 0, pruning_step 튜닝용)
#   --10  : 10 images main experiment (이미지 0~9)
#   --90  : 90 images final eval (이미지 10~99, --10과 합쳐서 100개)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

set -e

# 인자 파싱
RUN_1=false
RUN_10=false
RUN_90=false

for arg in "$@"; do
    case $arg in
        --1) RUN_1=true ;;
        --10) RUN_10=true ;;
        --90) RUN_90=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

if [ "$RUN_1" = false ] && [ "$RUN_10" = false ] && [ "$RUN_90" = false ]; then
    echo "사용법: bash exp2_pruning.sh [--1] [--10] [--90]"
    echo "  --1   : 1 image sanity check (이미지 0)"
    echo "  --10  : 10 images main experiment (이미지 0~9)"
    echo "  --90  : 90 images final eval (이미지 10~99, --10과 합쳐서 100개)"
    exit 0
fi

# ============================================================
# Repulsion Hyperparameters (Exp1과 동일)
# - scale: 튜닝 중 (scale=50은 너무 강함, scale=0.1~1.0은 효과 없음)
#   → scale=10부터 시작, ratio_scaled_to_score 0.1~0.3 목표
# - sigma_break=1.0: σ ∈ [1,10] 구간만 ON (~30/50 step)
# - schedule=constant: 추가 decay 없음
# ============================================================
REPULSION_SCALE=10            # 튜닝 중: 10 → ratio 보고 5 또는 15로 조정
REPULSION_SIGMA_BREAK=1.0     # σ < 1.0에서 OFF
REPULSION_SCHEDULE="constant" # 추가 decay 없음
PRUNING_STEP=25               # 4→2 pruning at step 25

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
    repulsion_scale=${REPULSION_SCALE} \
    repulsion_sigma_break=${REPULSION_SIGMA_BREAK} \
    repulsion_schedule=${REPULSION_SCHEDULE} \
    pruning_step=${PRUNING_STEP} \
    optimization_step=-1 \
    data.end_id=1 \
    name=exp2_sanity_check_scale${REPULSION_SCALE}_prune${PRUNING_STEP} \
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
    repulsion_scale=${REPULSION_SCALE} \
    repulsion_sigma_break=${REPULSION_SIGMA_BREAK} \
    repulsion_schedule=${REPULSION_SCHEDULE} \
    pruning_step=${PRUNING_STEP} \
    optimization_step=-1 \
    data.end_id=10 \
    name=exp2_10img_scale${REPULSION_SCALE}_prune${PRUNING_STEP} \
    gpu=0
fi

# ============================================================
# [실험 2] Final Eval - 90 images (10~99, 앞 10개는 --10에서 이미 실행)
# ============================================================
if [ "$RUN_90" = true ]; then
    echo "========== [실험 2] 90 images final eval (10~99) =========="
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
    repulsion_scale=${REPULSION_SCALE} \
    repulsion_sigma_break=${REPULSION_SIGMA_BREAK} \
    repulsion_schedule=${REPULSION_SCHEDULE} \
    pruning_step=${PRUNING_STEP} \
    optimization_step=-1 \
    data.start_id=10 \
    data.end_id=100 \
    name=exp2_90img_scale${REPULSION_SCALE}_prune${PRUNING_STEP} \
    gpu=0
fi

echo "완료!"
