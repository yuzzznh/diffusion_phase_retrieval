#!/bin/bash
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# [실험 5] Final Evaluation - Best Setting으로 100 images
# 목표: 제일 잘 나온 세팅으로 최종 평가 및 비교 Table 생성
#
# 사용법: bash exp5_final.sh [--imagenet] [--ffhq]
#   --imagenet : ImageNet 90 images (이미지 10~99, 앞 10개는 다른 실험에서 사용)
#   --ffhq     : FFHQ 100 images (옵션 - 여건 안되면 생략 가능)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

set -e

# 인자 파싱
RUN_IMAGENET=false
RUN_FFHQ=false

for arg in "$@"; do
    case $arg in
        --imagenet) RUN_IMAGENET=true ;;
        --ffhq) RUN_FFHQ=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

if [ "$RUN_IMAGENET" = false ] && [ "$RUN_FFHQ" = false ]; then
    echo "사용법: bash exp5_final.sh [--imagenet] [--ffhq]"
    echo "  --imagenet : ImageNet 90 images (이미지 10~99)"
    echo "  --ffhq     : FFHQ 100 images (옵션)"
    exit 0
fi

# TODO exp 1/2/3에서 결정된 Hyperparameter 값을 가져와서 반영할 것!

# ============================================================
# Repulsion Hyperparameters
# NOTE: 실험 1~4 결과 보고 best hyperparameter로 업데이트
# - scale: 튜닝 중 (scale=50은 너무 강함, scale=0.1~1.0은 효과 없음)
#   → scale=10부터 시작, ratio_scaled_to_score 0.1~0.3 목표
# - sigma_break=1.0: σ ∈ [1,10] 구간만 ON (~30/50 step)
# - schedule=constant: 추가 decay 없음
# ============================================================
REPULSION_SCALE=10            # 튜닝 중: 10 → ratio 보고 5 또는 15로 조정
REPULSION_SIGMA_BREAK=1.0     # σ < 1.0에서 OFF
REPULSION_SCHEDULE="constant" # 추가 decay 없음
PRUNING_STEP=25               # 4→2 pruning at step 25
OPTIMIZATION_STEP=25          # latent optimization from step 25

# ============================================================
# [실험 5] ImageNet 90 images - Best Setting (10~99, 앞 10개는 다른 실험에서 사용)
# ============================================================
if [ "$RUN_IMAGENET" = true ]; then
    echo "========== [실험 5] ImageNet 90 images (10~99) =========="
    python posterior_sample.py \
    +data=test-imagenet \
    +model=imagenet256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/exp5_final/imagenet_100img \
    num_samples=4 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=${REPULSION_SCALE} \
    repulsion_sigma_break=${REPULSION_SIGMA_BREAK} \
    repulsion_schedule=${REPULSION_SCHEDULE} \
    pruning_step=${PRUNING_STEP} \
    optimization_step=${OPTIMIZATION_STEP} \
    data.start_id=10 \
    data.end_id=100 \
    name=exp5_imagenet_90img_scale${REPULSION_SCALE}_prune${PRUNING_STEP}_opt${OPTIMIZATION_STEP} \
    gpu=0
fi

# ============================================================
# [실험 5] FFHQ 100 images - Best Setting
# NOTE: 시간이 남으면 실행, 안 되면 "ImageNet이 상위 호환 문제이므로 생략" 가능
# ============================================================
if [ "$RUN_FFHQ" = true ]; then
    echo "========== [실험 5] FFHQ 100 images =========="
    python posterior_sample.py \
    +data=test-ffhq \
    +model=ffhq256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/exp5_final/ffhq_100img \
    num_samples=4 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=${REPULSION_SCALE} \
    repulsion_sigma_break=${REPULSION_SIGMA_BREAK} \
    repulsion_schedule=${REPULSION_SCHEDULE} \
    pruning_step=${PRUNING_STEP} \
    optimization_step=${OPTIMIZATION_STEP} \
    data.end_id=100 \
    name=exp5_ffhq_100img_scale${REPULSION_SCALE}_prune${PRUNING_STEP}_opt${OPTIMIZATION_STEP} \
    gpu=0
fi

echo "완료!"
