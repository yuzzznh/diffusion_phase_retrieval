#!/bin/bash
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# [실험 4] Hard Data Consistency Optimization (ReSample-style)
# 목표: 실험 1~3 중 가장 잘 나온 세팅에 latent optimization 추가
# 확인 지표: PSNR 향상, Optimization 횟수/소요시간
#
# 사용법: bash exp4_optimization.sh [--1] [--10] [--100]
#   --1   : 1 image sanity check (optimization_step 튜닝용)
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
    echo "사용법: bash exp4_optimization.sh [--1] [--10] [--100]"
    echo "  --1   : 1 image sanity check"
    echo "  --10  : 10 images main experiment"
    echo "  --100 : 100 images final eval"
    exit 0
fi

# ============================================================
# [실험 4] Sanity Check - 1 image
# TODO: optimization_step 튜닝 후 값 변경
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
    repulsion_scale=0.1 \
    pruning_step=25 \
    optimization_step=25 \
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
    repulsion_scale=0.1 \
    pruning_step=25 \
    optimization_step=25 \
    data.end_id=10 \
    name=exp4_10img \
    gpu=0
fi

# ============================================================
# [실험 4] Final Eval - 100 images
# ============================================================
if [ "$RUN_100" = true ]; then
    echo "========== [실험 4] 100 images final eval =========="
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
    repulsion_scale=0.1 \
    pruning_step=25 \
    optimization_step=25 \
    data.end_id=100 \
    name=exp4_100img \
    gpu=0
fi

echo "완료!"
