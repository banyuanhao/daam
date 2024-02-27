# mean activation value of the feature maps

import argparse
from daam import trace, set_seed
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
import wandb
import random
import numpy as np
import json
    
#
parser = argparse.ArgumentParser(description='Diffusion')
parser.add_argument('--prompt', type=str, required=True)
parser.add_argument('--negative_prompt', type=str, default='')
parser.add_argument('--seed', type=int,nargs='+', default=[0])
parser.add_argument('--project', type=str, default='negative prompt')
parser.add_argument('--note', type=str, default='negative prompt')
parser.add_argument('--group', type=str, required=True)
parser.add_argument('--tags', metavar='S', type=str, nargs='+',default='negative prompt')
parser.add_argument('--steps', type=int, default=30)
parser.add_argument('--wandb',action='store_true',help='use wandb')

args = parser.parse_args()

wandb.login()

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

prompt = args.prompt
negative_prompt = args.negative_prompt
seeds = args.seed
steps = args.steps
tags = args.tags



if args.wandb:
    wandb.login()
    wandb.config = {"prompt": prompt,
                    "negative prompt": negative_prompt, 
                    "seeds": seeds,
                    "steps": steps,
                    "tags": tags,
                    }
    run = wandb.init(
        project=args.project,
        notes=args.note,
        group=args.group,
        config=wandb.config,
    )

ratio_list = []

for seed in iter(seeds):
    with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
        with trace(pipe) as tc:
            
            out = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=steps, generator=set_seed(seed),output_type='latent')
            
            pos, neg = tc.compute_activation_ratio()
            ratio = [neg[i]/pos[i] for i in range(len(pos))]
            
            ratio_list.append(ratio)


with open('wrapupdata/ratio_adj.json','r') as f:
    data = json.load(f)
    
data.append({'prompt':prompt, 'negative_prompt':negative_prompt, 'seeds':seeds, 'steps':steps, 'ratio':ratio_list})

with open('wrapupdata/ratio_adj.json', 'w') as f:
    json.dump(data, f)