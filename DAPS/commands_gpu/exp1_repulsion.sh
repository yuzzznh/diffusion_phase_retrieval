#!/bin/bash
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# [실험 1] 4-Particle Full Run (Repulsion vs. Independence)
# 목표: Repulsion ON vs DAPS Baseline 비교 - RLSD style SVGD repulsion in DINO feature space
# 확인 지표: Max PSNR, Std / Mode Coverage, Mean Pairwise Distance
#
# Repulsion Config:
#   - repulsion_scale=50: RLSD gamma=50 (HDR task) 기준
#   - repulsion_sigma_break=1.0: σ < 1.0에서 OFF → σ ∈ [1,10] 구간만 ON (~30/50 step)
#     (RLSD는 보통 더 오래 켜둠. 더 긴 ON 원하면 0.1 또는 0.01로 낮추기)
#   - repulsion_schedule='constant': 추가 decay 없음
#     (EDM score-ε 변환에 의해 ε 관점에서 σ가 곱해지는 효과 → RLSD gamma×sigma와 유사)
#
# 사용법: bash exp1_repulsion.sh [--1] [--10] [--90]
#   --1   : 1 image sanity check (이미지 0, repulsion_scale 튜닝용)
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
    echo "사용법: bash exp1_repulsion.sh [--1] [--10] [--90]"
    echo "  --1   : 1 image sanity check (이미지 0)"
    echo "  --10  : 10 images main experiment (이미지 0~9)"
    echo "  --90  : 90 images final eval (이미지 10~99, --10과 합쳐서 100개)"
    exit 0
fi

# ============================================================
# Repulsion Hyperparameters
# - scale: 튜닝 중 (scale=50은 너무 강함, scale=0.1~1.0은 효과 없음)
#   → scale=10부터 시작, ratio_scaled_to_score 0.1~0.3 목표
# - sigma_break=1.0: σ ∈ [1,10] 구간만 ON (~30/50 step)
# - schedule=constant: 추가 decay 없
# ============================================================
REPULSION_SCALE=10            # 튜닝 중: 10 → ratio 보고 5 또는 15로 조정
REPULSION_SIGMA_BREAK=1.0     # σ < 1.0에서 OFF (더 긴 ON: 0.1 또는 0.01)
REPULSION_SCHEDULE="constant" # 추가 decay 없음 (σ-decay는 score→ε 변환에서 자연 발생)

# ============================================================
# [실험 1] Sanity Check - 1 image
# 목적: repulsion이 정상 작동하는지 확인, pairwise distance 증가 확인
# ============================================================
if [ "$RUN_1" = true ]; then
    echo "========== [실험 1] 1 image sanity check =========="
    echo "Repulsion Config: scale=${REPULSION_SCALE}, sigma_break=${REPULSION_SIGMA_BREAK}, schedule=${REPULSION_SCHEDULE}"
    python posterior_sample.py \
    +data=test-imagenet \
    +model=imagenet256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/exp1_repulsion/imagenet_1img \
    num_samples=4 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=${REPULSION_SCALE} \
    repulsion_sigma_break=${REPULSION_SIGMA_BREAK} \
    repulsion_schedule=${REPULSION_SCHEDULE} \
    pruning_step=-1 \
    optimization_step=-1 \
    data.end_id=1 \
    name=exp1_sanity_check_scale${REPULSION_SCALE} \
    gpu=0
fi

# ============================================================
# [실험 1] Main Experiment - 10 images
# ============================================================
if [ "$RUN_10" = true ]; then
    echo "========== [실험 1] 10 images main experiment =========="
    echo "Repulsion Config: scale=${REPULSION_SCALE}, sigma_break=${REPULSION_SIGMA_BREAK}, schedule=${REPULSION_SCHEDULE}"
    python posterior_sample.py \
    +data=test-imagenet \
    +model=imagenet256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/exp1_repulsion/imagenet_10img \
    num_samples=4 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=${REPULSION_SCALE} \
    repulsion_sigma_break=${REPULSION_SIGMA_BREAK} \
    repulsion_schedule=${REPULSION_SCHEDULE} \
    pruning_step=-1 \
    optimization_step=-1 \
    data.end_id=10 \
    name=exp1_10img_scale${REPULSION_SCALE} \
    gpu=0
fi

# ============================================================
# [실험 1] Final Eval - 90 images (10~99, 앞 10개는 --10에서 이미 실행)
# ============================================================
if [ "$RUN_90" = true ]; then
    echo "========== [실험 1] 90 images final eval (10~99) =========="
    echo "Repulsion Config: scale=${REPULSION_SCALE}, sigma_break=${REPULSION_SIGMA_BREAK}, schedule=${REPULSION_SCHEDULE}"
    python posterior_sample.py \
    +data=test-imagenet \
    +model=imagenet256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/exp1_repulsion/imagenet_100img \
    num_samples=4 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=${REPULSION_SCALE} \
    repulsion_sigma_break=${REPULSION_SIGMA_BREAK} \
    repulsion_schedule=${REPULSION_SCHEDULE} \
    pruning_step=-1 \
    optimization_step=-1 \
    data.start_id=10 \
    data.end_id=100 \
    name=exp1_90img_scale${REPULSION_SCALE} \
    gpu=0
fi

echo "완료!"
