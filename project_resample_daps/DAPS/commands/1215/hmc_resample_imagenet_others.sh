# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This file contains the commands to run the LDM experiment on ImageNet dataset with DAPS 100 + ReSample.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ++++ Nonlinear Tasks ++++
# nonlinear deblur
(python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ldm \
+task=nonlinear_blur \
+sampler=latent_edm_daps \
task_group=ldm \
save_dir=results/hmc_resample_independent/imagenet \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=4 \
name=nonlinear_blur \
gpu=0 \
resample=True)

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
batch_size=4 \
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
batch_size=4 \
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
batch_size=4 \
name=gaussian_blur \
gpu=0 \
resample=True)

# motion blur
(python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ldm \
+task=motion_blur \
+sampler=latent_edm_daps \
task_group=ldm \
save_dir=results/hmc_resample_independent/imagenet \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=4 \
name=motion_blur \
gpu=0 \
resample=True)

# box inpainting
(python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ldm \
+task=inpainting \
+sampler=latent_edm_daps \
task_group=ldm \
save_dir=results/hmc_resample_independent/imagenet \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=4 \
name=inpainting \
gpu=0 \
resample=True)

# random inpainting
(python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ldm \
+task=inpainting_rand \
+sampler=latent_edm_daps \
task_group=ldm \
save_dir=results/hmc_resample_independent/imagenet \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=4 \
name=inpainting_rand \
gpu=0 \
resample=True)
