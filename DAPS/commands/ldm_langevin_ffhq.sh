# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This file contains the commands to run the LDM (Langevin) experiment on FFHQ dataset with DAPS 100.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# phase retrieval
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ldm \
+task=phase_retrieval \
+sampler=latent_edm_daps \
task_group=ldm_langevin \
save_dir=results/ldm_langevin/ffhq \
num_samples=4 \
sampler.diffusion_scheduler_config.num_steps=2 \
sampler.annealing_scheduler_config.num_steps=50 \
name=phase_retrieval \
gpu=0
