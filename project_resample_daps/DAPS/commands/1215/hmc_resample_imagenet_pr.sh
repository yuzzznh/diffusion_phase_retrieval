# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This file contains the commands to run the LDM experiment on ImageNet dataset with DAPS 100 + ReSample.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ++++ Nonlinear Tasks ++++
# phase retrieval
python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ldm \
+task=phase_retrieval \
+sampler=latent_edm_daps \
task_group=ldm \
save_dir=results/hmc_resample_independent/imagenet \
num_runs=4 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
batch_size=4 \
name=phase_retrieval \
gpu=0 \
resample=True
