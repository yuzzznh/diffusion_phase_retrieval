# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# LatentDAPS (Langevin) - ImageNet Phase Retrieval
# 실험 0~5 메인 command 파일
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ============================================================
# [실험 0] DAPS Baseline - 1 image (Sanity Check)
# ============================================================
# python posterior_sample.py \
# +data=test-imagenet \
# +model=imagenet256ldm \
# +task=phase_retrieval \
# +sampler=latent_edm_daps \
# task_group=ldm_langevin \
# save_dir=results/exp0_baseline/imagenet_1img \
# num_samples=4 \
# sampler.diffusion_scheduler_config.num_steps=2 \
# sampler.annealing_scheduler_config.num_steps=50 \
# repulsion_scale=0.0 \
# pruning_step=-1 \
# optimization_step=-1 \
# data.end_id=1 \
# name=exp0_sanity_check \
# gpu=0

# ============================================================
# [실험 0] DAPS Baseline - 10 images (Main Experiment)
# ============================================================
# python posterior_sample.py \
# +data=test-imagenet \
# +model=imagenet256ldm \
# +task=phase_retrieval \
# +sampler=latent_edm_daps \
# task_group=ldm_langevin \
# save_dir=results/exp0_baseline/imagenet_10img \
# num_samples=4 \
# sampler.diffusion_scheduler_config.num_steps=2 \
# sampler.annealing_scheduler_config.num_steps=50 \
# repulsion_scale=0.0 \
# pruning_step=-1 \
# optimization_step=-1 \
# data.end_id=10 \
# name=exp0_10img \
# gpu=0

# ============================================================
# [실험 0] DAPS Baseline - 100 images (Final Eval)
# ============================================================
# python posterior_sample.py \
# +data=test-imagenet \
# +model=imagenet256ldm \
# +task=phase_retrieval \
# +sampler=latent_edm_daps \
# task_group=ldm_langevin \
# save_dir=results/exp0_baseline/imagenet_100img \
# num_samples=4 \
# sampler.diffusion_scheduler_config.num_steps=2 \
# sampler.annealing_scheduler_config.num_steps=50 \
# repulsion_scale=0.0 \
# pruning_step=-1 \
# optimization_step=-1 \
# data.end_id=100 \
# name=exp0_100img \
# gpu=0

# ============================================================
# [실험 1] Repulsion Only - 10 images (TODO: repulsion_scale 튜닝 후 값 변경)
# ============================================================
# python posterior_sample.py \
# +data=test-imagenet \
# +model=imagenet256ldm \
# +task=phase_retrieval \
# +sampler=latent_edm_daps \
# task_group=ldm_langevin \
# save_dir=results/exp1_repulsion/imagenet_10img \
# num_samples=4 \
# sampler.diffusion_scheduler_config.num_steps=2 \
# sampler.annealing_scheduler_config.num_steps=50 \
# repulsion_scale=0.1 \
# pruning_step=-1 \
# optimization_step=-1 \
# data.end_id=10 \
# name=exp1_10img \
# gpu=0

# ============================================================
# [실험 2] Pruning (4->2) - 10 images (TODO: pruning_step 튜닝 후 값 변경)
# ============================================================
# python posterior_sample.py \
# +data=test-imagenet \
# +model=imagenet256ldm \
# +task=phase_retrieval \
# +sampler=latent_edm_daps \
# task_group=ldm_langevin \
# save_dir=results/exp2_pruning/imagenet_10img \
# num_samples=4 \
# sampler.diffusion_scheduler_config.num_steps=2 \
# sampler.annealing_scheduler_config.num_steps=50 \
# repulsion_scale=0.1 \
# pruning_step=25 \
# optimization_step=-1 \
# data.end_id=10 \
# name=exp2_10img \
# gpu=0

# ============================================================
# [실험 3] 2-Particle Full Run - 10 images
# ============================================================
# python posterior_sample.py \
# +data=test-imagenet \
# +model=imagenet256ldm \
# +task=phase_retrieval \
# +sampler=latent_edm_daps \
# task_group=ldm_langevin \
# save_dir=results/exp3_2particle/imagenet_10img \
# num_samples=2 \
# sampler.diffusion_scheduler_config.num_steps=2 \
# sampler.annealing_scheduler_config.num_steps=50 \
# repulsion_scale=0.1 \
# pruning_step=-1 \
# optimization_step=-1 \
# data.end_id=10 \
# name=exp3_10img \
# gpu=0

# ============================================================
# [실험 4] Hard Data Consistency Optimization - 10 images (TODO: optimization_step 튜닝 후 값 변경)
# ============================================================
# python posterior_sample.py \
# +data=test-imagenet \
# +model=imagenet256ldm \
# +task=phase_retrieval \
# +sampler=latent_edm_daps \
# task_group=ldm_langevin \
# save_dir=results/exp4_optimization/imagenet_10img \
# num_samples=4 \
# sampler.diffusion_scheduler_config.num_steps=2 \
# sampler.annealing_scheduler_config.num_steps=50 \
# repulsion_scale=0.1 \
# pruning_step=25 \
# optimization_step=25 \
# data.end_id=10 \
# name=exp4_10img \
# gpu=0
