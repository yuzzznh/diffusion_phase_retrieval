#!/bin/bash
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# [실험 1] 4-Particle Full Run (Repulsion vs. Independence)
# 목표: Repulsion ON vs DAPS Baseline 비교
# 확인 지표: Max PSNR, Std / Mode Coverage
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
# [실험 1] Sanity Check - 1 image
# TODO: repulsion_scale 튜닝 후 값 변경
# ============================================================
if [ "$RUN_1" = true ]; then
    echo "========== [실험 1] 1 image sanity check =========="
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
    repulsion_scale=0.1 \
    pruning_step=-1 \
    optimization_step=-1 \
    data.end_id=1 \
    name=exp1_sanity_check \
    gpu=0
fi

# ============================================================
# [실험 1] Main Experiment - 10 images
# ============================================================
if [ "$RUN_10" = true ]; then
    echo "========== [실험 1] 10 images main experiment =========="
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
    repulsion_scale=0.1 \
    pruning_step=-1 \
    optimization_step=-1 \
    data.end_id=10 \
    name=exp1_10img \
    gpu=0
fi

# ============================================================
# [실험 1] Final Eval - 90 images (10~99, 앞 10개는 --10에서 이미 실행)
# ============================================================
if [ "$RUN_90" = true ]; then
    echo "========== [실험 1] 90 images final eval (10~99) =========="
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
    repulsion_scale=0.1 \
    pruning_step=-1 \
    optimization_step=-1 \
    data.start_id=10 \
    data.end_id=100 \
    name=exp1_90img \
    gpu=0
fi

echo "완료!"
