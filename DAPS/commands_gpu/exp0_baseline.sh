#!/bin/bash
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# [실험 0] DAPS Baseline (LatentDAPS with Langevin Dynamic)
# 목표: LatentDAPS의 ImageNet Phase Retrieval 성능 측정 (Reference)
# 설정: num_samples=4, repulsion_scale=0.0 (독립 실행 = DAPS 4 runs)
#
# 사용법: bash exp0_baseline.sh [--1] [--10] [--90]
#   --1   : 1 image sanity check (이미지 0)
#   --10  : 10 images main experiment (이미지 0~9)
#   --90  : 90 images final eval (이미지 10~99, --10과 합쳐서 100개)
#   (인자 없으면 사용법 출력)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

set -e  # 에러 발생 시 중단

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

# 인자 없으면 사용법 출력
if [ "$RUN_1" = false ] && [ "$RUN_10" = false ] && [ "$RUN_90" = false ]; then
    echo "사용법: bash exp0_baseline.sh [--1] [--10] [--90]"
    echo "  --1   : 1 image sanity check (이미지 0)"
    echo "  --10  : 10 images main experiment (이미지 0~9)"
    echo "  --90  : 90 images final eval (이미지 10~99, --10과 합쳐서 100개)"
    exit 0
fi

# ============================================================
# [실험 0] Sanity Check - 1 image
# ============================================================
if [ "$RUN_1" = true ]; then
    echo "========== [실험 0] 1 image sanity check =========="
    python posterior_sample.py \
    +data=test-imagenet \
    +model=imagenet256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/exp0_baseline/imagenet_1img \
    num_samples=4 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=0.0 \
    pruning_step=-1 \
    hard_data_consistency=-1 \
    data.end_id=1 \
    name=exp0_sanity_check \
    gpu=0
fi

# ============================================================
# [실험 0] Main Experiment - 10 images
# ============================================================
if [ "$RUN_10" = true ]; then
    echo "========== [실험 0] 10 images main experiment =========="
    python posterior_sample.py \
    +data=test-imagenet \
    +model=imagenet256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/exp0_baseline/imagenet_10img \
    num_samples=4 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=0.0 \
    pruning_step=-1 \
    hard_data_consistency=-1 \
    data.end_id=10 \
    name=exp0_10img \
    gpu=0
fi

# ============================================================
# [실험 0] Final Eval - 90 images (10~99, 앞 10개는 --10에서 이미 실행)
# ============================================================
if [ "$RUN_90" = true ]; then
    echo "========== [실험 0] 90 images final eval (10~99) =========="
    python posterior_sample.py \
    +data=test-imagenet \
    +model=imagenet256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/exp0_baseline/imagenet_100img \
    num_samples=4 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=0.0 \
    pruning_step=-1 \
    hard_data_consistency=-1 \
    data.start_id=10 \
    data.end_id=100 \
    name=exp0_90img \
    gpu=0
fi

echo "완료!"
