#!/bin/bash
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# [실험 3] 2-Particle Full Run (Justification for '4') - TPU 버전
# 목표: "처음부터 2개만 돌리면 안 돼?" 질문에 대한 답변
# 확인 지표: Success Rate - Exp 2 (4→2)보다 낮아야 함
#
# 사용법: bash exp3_2particle.sh [--10] [--100]
#   --10  : 10 images main experiment
#   --100 : 100 images final eval
#   (1 image는 의미 없음 - 실패 비율을 재야 하므로)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

set -e

# TPU 환경변수 설정 (서브쉘에서도 동작하도록)
export PJRT_DEVICE=TPU
export PJRT_SELECT_DEFAULT_DEVICE=1

# 인자 파싱
RUN_10=false
RUN_100=false

for arg in "$@"; do
    case $arg in
        --10) RUN_10=true ;;
        --100) RUN_100=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

if [ "$RUN_10" = false ] && [ "$RUN_100" = false ]; then
    echo "사용법: bash exp3_2particle.sh [--10] [--100]"
    echo "  --10  : 10 images main experiment"
    echo "  --100 : 100 images final eval"
    echo "  (1 image는 의미 없음 - 실패 비율을 재야 함)"
    exit 0
fi

# ============================================================
# [실험 3] Main Experiment - 10 images (TPU)
# ============================================================
if [ "$RUN_10" = true ]; then
    echo "========== [실험 3] 10 images main experiment (TPU) =========="
    python posterior_sample.py \
    +data=test-imagenet \
    +model=imagenet256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/exp3_2particle_tpu/imagenet_10img \
    num_samples=2 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=0.1 \
    pruning_step=-1 \
    optimization_step=-1 \
    data.end_id=10 \
    name=exp3_10img_tpu \
    use_tpu=true
fi

# ============================================================
# [실험 3] Final Eval - 100 images (TPU)
# ============================================================
if [ "$RUN_100" = true ]; then
    echo "========== [실험 3] 100 images final eval (TPU) =========="
    python posterior_sample.py \
    +data=test-imagenet \
    +model=imagenet256ldm \
    +task=phase_retrieval \
    +sampler=latent_edm_daps \
    task_group=ldm_langevin \
    save_dir=results/exp3_2particle_tpu/imagenet_100img \
    num_samples=2 \
    sampler.diffusion_scheduler_config.num_steps=2 \
    sampler.annealing_scheduler_config.num_steps=50 \
    repulsion_scale=0.1 \
    pruning_step=-1 \
    optimization_step=-1 \
    data.end_id=100 \
    name=exp3_100img_tpu \
    use_tpu=true
fi

echo "완료!"
