#!/bin/bash
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# [실험 1] 4-Particle Full Run (Repulsion vs. Independence) - TPU 버전
# 목표: Repulsion ON vs DAPS Baseline 비교
# 확인 지표: Max PSNR, Std / Mode Coverage
#
# 사용법: bash exp1_repulsion.sh [--1] [--10] [--100]
#   --1   : 1 image sanity check (repulsion_scale 튜닝용)
#   --10  : 10 images main experiment
#   --100 : 100 images final eval
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

set -e

# TPU 환경변수 설정 (서브쉘에서도 동작하도록)
export PJRT_DEVICE=TPU
export PJRT_SELECT_DEFAULT_DEVICE=1

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
    echo "사용법: bash exp1_repulsion.sh [--1] [--10] [--100]"
    echo "  --1   : 1 image sanity check"
    echo "  --10  : 10 images main experiment"
    echo "  --100 : 100 images final eval"
    exit 0
fi

# ============================================================
# [실험 1] Sanity Check - 1 image (TPU)
# TODO: repulsion_scale 튜닝 후 값 변경
# ============================================================
if [ "$RUN_1" = true ]; then
    echo "========== [실험 1] 1 image sanity check (TPU) =========="
    python posterior_sample.py \
    +data=test-imagenet \
    +model=imagenet256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/exp1_repulsion_tpu/imagenet_1img \
    num_samples=4 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=0.1 \
    pruning_step=-1 \
    optimization_step=-1 \
    data.end_id=1 \
    name=exp1_sanity_check_tpu \
    use_tpu=true
fi

# ============================================================
# [실험 1] Main Experiment - 10 images (TPU)
# ============================================================
if [ "$RUN_10" = true ]; then
    echo "========== [실험 1] 10 images main experiment (TPU) =========="
    python posterior_sample.py \
    +data=test-imagenet \
    +model=imagenet256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/exp1_repulsion_tpu/imagenet_10img \
    num_samples=4 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=0.1 \
    pruning_step=-1 \
    optimization_step=-1 \
    data.end_id=10 \
    name=exp1_10img_tpu \
    use_tpu=true
fi

# ============================================================
# [실험 1] Final Eval - 100 images (TPU)
# ============================================================
if [ "$RUN_100" = true ]; then
    echo "========== [실험 1] 100 images final eval (TPU) =========="
    python posterior_sample.py \
    +data=test-imagenet \
    +model=imagenet256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/exp1_repulsion_tpu/imagenet_100img \
    num_samples=4 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=0.1 \
    pruning_step=-1 \
    optimization_step=-1 \
    data.end_id=100 \
    name=exp1_100img_tpu \
    use_tpu=true
fi

echo "완료!"
