#!/bin/bash
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# [실험 4] Hard Data Consistency Optimization (ReSample-style)
# 목표: 실험 1~3 중 가장 잘 나온 세팅(실험 1, scale=10)에 latent optimization 추가
# 확인 지표: PSNR 향상, Optimization 횟수/소요시간, 각 batch element의 independent termination
#
# 구현 요약:
# - 맨 마지막 timestep에서만 optimization 수행 (diffusion loop 완료 후)
# - Loss: || A(decode(z)) - y ||^2 (measurement MSE)
# - Termination: (1) cur_loss < eps² (1e-6), (2) 200 iter 후 init_loss < cur_loss
# - **Batch element 간 optimization & termination이 independent** (ReSample 공식 레포와 다름!)
#
# 사용법: bash exp4_optimization.sh [--1] [--10] [--90]
#   --1   : 1 image sanity check (이미지 0)
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
    echo "사용법: bash exp4_optimization.sh [--1] [--10] [--90]"
    echo "  --1   : 1 image sanity check (이미지 0)"
    echo "  --10  : 10 images main experiment (이미지 0~9)"
    echo "  --90  : 90 images final eval (이미지 10~99, --10과 합쳐서 100개)"
    exit 0
fi

# ============================================================
# Hyperparameters: 실험 1 세팅 (scale=10, Sweet Spot) + Optimization
# - Repulsion: scale=10, sigma_break=1.0 (σ < 1.0에서 OFF), schedule=constant
# - Pruning: 비활성화 (-1) - 실험 4는 optimization 효과 검증이 목적
# - Optimization: lr=5e-3, eps=1e-3, max_iters=500 (ReSample defaults)
# ============================================================
REPULSION_SCALE=10
REPULSION_SIGMA_BREAK=1.0
REPULSION_SCHEDULE="constant"
PRUNING_STEP=-1               # 비활성화 (실험 4에서는 pruning 사용 안 함)
HARD_DATA_CONSISTENCY=1       # 1이면 on (맨 마지막에 latent optimization 수행), -1이면 off
OPTIMIZATION_LR=5e-3          # ReSample default
OPTIMIZATION_EPS=1e-3         # cur_loss < eps² → terminate
OPTIMIZATION_MAX_ITERS=500    # ReSample default

# ============================================================
# [실험 4] Sanity Check - 1 image
# ============================================================
if [ "$RUN_1" = true ]; then
    echo "========== [실험 4] 1 image sanity check =========="
    python posterior_sample.py \
    +data=test-imagenet \
    +model=imagenet256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/exp4_optimization/imagenet_1img \
    num_samples=4 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=${REPULSION_SCALE} \
    repulsion_sigma_break=${REPULSION_SIGMA_BREAK} \
    repulsion_schedule=${REPULSION_SCHEDULE} \
    pruning_step=${PRUNING_STEP} \
    hard_data_consistency=${HARD_DATA_CONSISTENCY} \
    optimization_lr=${OPTIMIZATION_LR} \
    optimization_eps=${OPTIMIZATION_EPS} \
    optimization_max_iters=${OPTIMIZATION_MAX_ITERS} \
    data.end_id=1 \
    name=exp4_sanity_check \
    gpu=0
fi

# ============================================================
# [실험 4] Main Experiment - 10 images
# ============================================================
if [ "$RUN_10" = true ]; then
    echo "========== [실험 4] 10 images main experiment =========="
    python posterior_sample.py \
    +data=test-imagenet \
    +model=imagenet256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/exp4_optimization/imagenet_10img \
    num_samples=4 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=${REPULSION_SCALE} \
    repulsion_sigma_break=${REPULSION_SIGMA_BREAK} \
    repulsion_schedule=${REPULSION_SCHEDULE} \
    pruning_step=${PRUNING_STEP} \
    hard_data_consistency=${HARD_DATA_CONSISTENCY} \
    optimization_lr=${OPTIMIZATION_LR} \
    optimization_eps=${OPTIMIZATION_EPS} \
    optimization_max_iters=${OPTIMIZATION_MAX_ITERS} \
    data.end_id=10 \
    name=exp4_10img \
    gpu=0
fi

# ============================================================
# [실험 4] Final Eval - 90 images (10~99, 앞 10개는 --10에서 이미 실행)
# ============================================================
if [ "$RUN_90" = true ]; then
    echo "========== [실험 4] 90 images final eval (10~99) =========="
    python posterior_sample.py \
    +data=test-imagenet \
    +model=imagenet256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/exp4_optimization/imagenet_100img \
    num_samples=4 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=${REPULSION_SCALE} \
    repulsion_sigma_break=${REPULSION_SIGMA_BREAK} \
    repulsion_schedule=${REPULSION_SCHEDULE} \
    pruning_step=${PRUNING_STEP} \
    hard_data_consistency=${HARD_DATA_CONSISTENCY} \
    optimization_lr=${OPTIMIZATION_LR} \
    optimization_eps=${OPTIMIZATION_EPS} \
    optimization_max_iters=${OPTIMIZATION_MAX_ITERS} \
    data.start_id=10 \
    data.end_id=100 \
    name=exp4_90img \
    gpu=0
fi

echo "완료!"
