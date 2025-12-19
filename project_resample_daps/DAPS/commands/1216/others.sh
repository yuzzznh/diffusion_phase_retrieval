# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This file contains the commands to run the LDM experiment on ImageNet dataset with DAPS 100 + ReSample.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ++++ Nonlinear Tasks ++++
# high dynamic range
(python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ldm \
+task=hdr \
+sampler=latent_edm_daps \
task_group=ldm \
save_dir=results/hmc_resample_independent/imagenet \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=10 \
name=hdr \
gpu=0 \
resample=True)

# ++++ Linear Tasks ++++
# down sampling
(python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ldm \
+task=down_sampling \
+sampler=latent_edm_daps \
task_group=ldm \
save_dir=results/hmc_resample_independent/imagenet \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=10 \
name=down_sampling \
gpu=0 \
resample=True)

# Gaussian blur
(python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ldm \
+task=gaussian_blur \
+sampler=latent_edm_daps \
task_group=ldm \
save_dir=results/hmc_resample_independent/imagenet \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=10 \
name=gaussian_blur \
gpu=0 \
resample=True)
