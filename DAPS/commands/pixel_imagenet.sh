# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This file contains the commands to run the pixel diffusion experiment on ImageNet dataset with DAPS 1K.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# phase retrieval
python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ddpm \
+task=phase_retrieval \
+sampler=edm_daps \
task_group=pixel \
save_dir=results/pixel/imagenet \
num_samples=4 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
name=phase_retrieval \
gpu=0
