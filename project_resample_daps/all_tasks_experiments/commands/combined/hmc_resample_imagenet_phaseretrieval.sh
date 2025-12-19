# phase retrieval
python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ldm \
+task=phase_retrieval \
+sampler=latent_edm_daps \
task_group=ldm \
save_dir=results/hmc_resample/imagenet \
num_runs=4 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=10 \
name=phase_retrieval \
gpu=0 \
resample=True

# # phase retrieval
# # 1장: L40S 기준 325/350W, 5GB/45GB, 98%. 7*4=28분 소요됨.
# # 4장: 310W 17GB 100% 39*4=156분
# python posterior_sample.py \
# +data=test-imagenet \
# +model=imagenet256ldm \
# +task=phase_retrieval \
# +sampler=latent_edm_daps \
# task_group=ldm \
# save_dir=results/hmc_and_vanilla_resample/imagenet \
# num_runs=4 \
# sampler.diffusion_scheduler_config.num_steps=2 \
# sampler.annealing_scheduler_config.num_steps=50 \
# batch_size=4 \
# name=phase_retrieval_1img \
# gpu=0 \
# resample=True \
# data.start_id=27 \
# data.end_id=31 
# # [start_id, end_id) 인덱싱. name도 바꿔줘야 파일 안 헷갈림 주의.