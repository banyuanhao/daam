import argparse
from daam import trace, set_seed
from diffusers import StableDiffusionPipeline
from models.diffuserpipeline import StableDiffusionPipelineForNegativePrompts
import torch
import matplotlib.pyplot as plt
import os
import math
import random
from pathlib import Path

seed = 53725
class_names = 'handbag.txt'

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)  

dataset_path = Path(f'dataset/ODFN/version_2')
prompts_path = dataset_path/'prompts_10000_5_5'
prompt_path = prompts_path/class_names
latents = pipe.get_latents(prompt='none', generator=set_seed(seed))

with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
    with trace(pipe) as tc:
        
        fig, axs = plt.subplots(2, 5, figsize=(5*5, 5*2))
        
        with open(prompt_path, 'r') as f:
            prompts = f.read()
            prompts = prompts.split('\n')
            name = prompts[0]
            prompts = prompts[1:]
            
        for k, prompt in enumerate(prompts):
            out = pipe(prompt=prompt, generator=set_seed(seed),latents = latents)
            heat_map = tc.compute_global_heat_map(time_idx=[0,1,2])
            heat_map_word = heat_map.compute_word_heat_map(class_names[:-4].replace('_', ' '))
            heat_map_word.plot_overlay_with_raw(out.images[0], axs[0][k])
        # flip the latent
        latents = latents.flip(3)
        for k, prompt in enumerate(prompts):
            out = pipe(prompt=prompt, generator=set_seed(seed),latents = latents)
            heat_map = tc.compute_global_heat_map(time_idx=[0,1,2])
            heat_map_word = heat_map.compute_word_heat_map(class_names[:-4].replace('_', ' '))
            heat_map_word.plot_overlay_with_raw(out.images[0], axs[0][k])
        fig.savefig(f'pics/daam_flip_{class_names[:-4]}_{seed}_img.png')