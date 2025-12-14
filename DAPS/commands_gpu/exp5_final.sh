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

# ============================================================
# Repulsion Hyperparameters
# NOTE: 실험 1~4 결과 보고 best hyperparameter로 업데이트
# - scale=50: RLSD gamma=50 (HDR task) 기준
# - sigma_break=1.0: σ ∈ [1,10] 구간만 ON (~30/50 step)
# - schedule=constant: 추가 decay 없음 (σ-decay는 score→ε 변환에서 자연 발생)
# ============================================================
REPULSION_SCALE=50            # RLSD gamma 기준
REPULSION_SIGMA_BREAK=1.0     # σ < 1.0에서 OFF
REPULSION_SCHEDULE="constant" # 추가 decay 없음

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
    pruning_step=25 \
    optimization_step=25 \
    data.start_id=10 \
    data.end_id=100 \
    name=exp5_imagenet_90img \
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
    pruning_step=25 \
    optimization_step=25 \
    data.end_id=100 \
    name=exp5_ffhq_100img \
    gpu=0
fi

echo "완료!"
