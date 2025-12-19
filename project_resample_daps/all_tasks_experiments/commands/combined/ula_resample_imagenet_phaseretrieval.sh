# phase retrieval
python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ldm \
+task=phase_retrieval \
+sampler=latent_edm_daps \
task_group=ldm_langevin \
save_dir=results/ula_resample/imagenet \
num_runs=4 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=10 \
name=phase_retrieval \
gpu=0 \
resample=True
