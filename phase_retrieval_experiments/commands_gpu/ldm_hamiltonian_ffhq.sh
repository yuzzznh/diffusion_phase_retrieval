# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# LatentDAPS (Hamiltonian) - FFHQ Phase Retrieval
# NOTE: 현재 프로젝트에서는 Langevin만 사용 (이 파일은 참고용)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ============================================================
# DAPS Baseline (Hamiltonian) - 100 images
# ============================================================
# python posterior_sample.py \
# +data=test-ffhq \
# +model=ffhq256ldm \
# +task=phase_retrieval \
# +sampler=latent_edm_daps \
# task_group=ldm_hamiltonian \
# save_dir=results/ldm_hamiltonian/ffhq_100img \
# num_samples=4 \
# sampler.diffusion_scheduler_config.num_steps=2 \
# sampler.annealing_scheduler_config.num_steps=50 \
# repulsion_scale=0.0 \
# pruning_step=-1 \
# optimization_step=-1 \
# data.end_id=100 \
# name=hamiltonian_ffhq_100img \
# gpu=0
