#!/bin/bash
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# [실험 3] 2-Particle Full Run (Justification for '4')
# 목표: "처음부터 2개만 돌리면 안 돼?" 질문에 대한 답변
# 확인 지표: Success Rate - Exp 2 (4→2)보다 낮아야 함
#
# 중요: N=2에서 SVGD bandwidth 계산이 정상 작동하는지 확인 필요 (RLSD 버그 수정됨)
# - h = median(dist)^2 / max(log(N), eps) 사용 (log(N-1)이 아님!)
#
# 사용법: bash exp3_2particle.sh [--1] [--10] [--90]
#   --1   : 1 image sanity check (N=2 bandwidth 버그 수정 확인용)
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
    echo "사용법: bash exp3_2particle.sh [--1] [--10] [--90]"
    echo "  --1   : 1 image sanity check (N=2 bandwidth 버그 수정 확인)"
    echo "  --10  : 10 images main experiment (이미지 0~9)"
    echo "  --90  : 90 images final eval (이미지 10~99, --10과 합쳐서 100개)"
    exit 0
fi

# ============================================================
# Repulsion Hyperparameters (Exp1과 동일)
# - scale=50: RLSD gamma=50 (HDR task) 기준
# - sigma_break=1.0: σ ∈ [1,10] 구간만 ON (~30/50 step)
# - schedule=constant: 추가 decay 없음 (σ-decay는 score→ε 변환에서 자연 발생)
# ============================================================
REPULSION_SCALE=50            # RLSD gamma 기준
REPULSION_SIGMA_BREAK=1.0     # σ < 1.0에서 OFF
REPULSION_SCHEDULE="constant" # 추가 decay 없음

# ============================================================
# [실험 3] Sanity Check - 1 image (N=2 bandwidth 버그 수정 확인)
# 목적: N=2에서 NaN이나 crash 없이 정상 작동하는지 확인
# ============================================================
if [ "$RUN_1" = true ]; then
    echo "========== [실험 3] 1 image sanity check (N=2) =========="
    echo "Repulsion Config: scale=${REPULSION_SCALE}, sigma_break=${REPULSION_SIGMA_BREAK}, schedule=${REPULSION_SCHEDULE}"
    python posterior_sample.py \
    +data=test-imagenet \
    +model=imagenet256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/exp3_2particle/imagenet_1img \
    num_samples=2 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=${REPULSION_SCALE} \
    repulsion_sigma_break=${REPULSION_SIGMA_BREAK} \
    repulsion_schedule=${REPULSION_SCHEDULE} \
    pruning_step=-1 \
    optimization_step=-1 \
    data.end_id=1 \
    name=exp3_sanity_check \
    gpu=0
fi

# ============================================================
# [실험 3] Main Experiment - 10 images
# ============================================================
if [ "$RUN_10" = true ]; then
    echo "========== [실험 3] 10 images main experiment =========="
    echo "Repulsion Config: scale=${REPULSION_SCALE}, sigma_break=${REPULSION_SIGMA_BREAK}, schedule=${REPULSION_SCHEDULE}"
    python posterior_sample.py \
    +data=test-imagenet \
    +model=imagenet256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/exp3_2particle/imagenet_10img \
    num_samples=2 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=${REPULSION_SCALE} \
    repulsion_sigma_break=${REPULSION_SIGMA_BREAK} \
    repulsion_schedule=${REPULSION_SCHEDULE} \
    pruning_step=-1 \
    optimization_step=-1 \
    data.end_id=10 \
    name=exp3_10img \
    gpu=0
fi

# ============================================================
# [실험 3] Final Eval - 90 images (10~99, 앞 10개는 --10에서 이미 실행)
# ============================================================
if [ "$RUN_90" = true ]; then
    echo "========== [실험 3] 90 images final eval (10~99) =========="
    echo "Repulsion Config: scale=${REPULSION_SCALE}, sigma_break=${REPULSION_SIGMA_BREAK}, schedule=${REPULSION_SCHEDULE}"
    python posterior_sample.py \
    +data=test-imagenet \
    +model=imagenet256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/exp3_2particle/imagenet_100img \
    num_samples=2 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=${REPULSION_SCALE} \
    repulsion_sigma_break=${REPULSION_SIGMA_BREAK} \
    repulsion_schedule=${REPULSION_SCHEDULE} \
    pruning_step=-1 \
    optimization_step=-1 \
    data.start_id=10 \
    data.end_id=100 \
    name=exp3_90img \
    gpu=0
fi

echo "완료!"
