# phase retrieval
python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ldm \
+task=phase_retrieval \
+sampler=latent_edm_daps \
task_group=ldm \
save_dir=results/hmc_resample_0to10_sequential/imagenet \
num_runs=4 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=10 \
name=phase_retrieval \
gpu=0 \
resample=True \
data.start_id=0 \
data.end_id=10
# [start_id, end_id) 인덱싱. name도 바꿔줘야 파일 안 헷갈림 주의.

# phase retrieval
python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ldm \
+task=phase_retrieval \
+sampler=latent_edm_daps \
task_group=ldm_langevin \
save_dir=results/ula_resample_0to10_sequential/imagenet \
num_runs=4 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=10 \
name=phase_retrieval \
gpu=0 \
resample=True \
data.start_id=0 \
data.end_id=10
# [start_id, end_id) 인덱싱. name도 바꿔줘야 파일 안 헷갈림 주의.
