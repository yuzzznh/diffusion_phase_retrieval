# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This file contains the commands to run the Stable Diffusion (SD v1.5) experiment on ImageNet dataset with DAPS 100.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# phase retrieval
python posterior_sample.py \
+data=test-imagenet \
+model=stable-diffusion-v1.5 \
+task=phase_retrieval \
+sampler=sd_edm_daps \
task_group=sd \
save_dir=results/sd/imagenet \
num_samples=4 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
name=phase_retrieval \
gpu=0
