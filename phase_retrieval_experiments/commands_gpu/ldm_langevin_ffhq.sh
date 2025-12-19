# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# LatentDAPS (Langevin) - FFHQ Phase Retrieval
# 실험 5 Final Eval용 (옵션 - 여건 안되면 생략)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ============================================================
# [실험 5] Final Eval - FFHQ 100 images (Best Setting 적용)
# ============================================================
# python posterior_sample.py \
# +data=test-ffhq \
# +model=ffhq256ldm \
# +task=phase_retrieval \
# +sampler=latent_edm_daps \
# task_group=ldm_langevin \
# save_dir=results/exp5_final/ffhq_100img \
# num_samples=4 \
# sampler.diffusion_scheduler_config.num_steps=2 \
# sampler.annealing_scheduler_config.num_steps=50 \
# repulsion_scale=0.0 \
# pruning_step=-1 \
# optimization_step=-1 \
# data.end_id=100 \
# name=exp5_ffhq_100img \
# gpu=0
