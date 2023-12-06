# a script to test the changing of the negative prompt
import argparse
from daam import set_seed
from models.diffuserpipeline import StableDiffusionPipelineForNegativePrompts
import torch
import matplotlib.pyplot as plt
import os
import wandb
import math
import random

def get_plt(num):
    plt.clf()
    plt.rcParams.update({'font.size': 10})
    if num < 5:
        fig, axs = plt.subplots(1, num , figsize=(5*num, 5+1))
        for ax in axs:
            ax.axis('off')
    else:
        fig, axs = plt.subplots(math.ceil(num / 4),4)

        for i in range(math.ceil(num / 4)):
            for j in range(4):
                ax = axs[i][j]
                ax.axis('off')
    plt.subplots_adjust(top=0.95)  
    return plt, fig, axs
    

parser = argparse.ArgumentParser(description='Diffusion')
parser.add_argument('--prompt', type=str, required=True)
parser.add_argument('--negative_prompt', type=str, default='')

parser.add_argument('--seed', type=int,nargs='+', default=[0])
parser.add_argument('--negative_time', type=int, nargs='+', default=None)

parser.add_argument('--project', type=str, default='negative prompt')
parser.add_argument('--note', type=str, default='negative prompt')
parser.add_argument('--group', type=str, required=True)
parser.add_argument('--tags', metavar='S', type=str, nargs='+',default='negative prompt')
parser.add_argument('--steps', type=int, default=30)
parser.add_argument('--wandb',action='store_true',help='use wandb')
args = parser.parse_args()

#os.environ["WANDB_MODE"] = "offline"
wandb.login()

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipelineForNegativePrompts.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

prompt = args.prompt
negative_prompt = args.negative_prompt
seeds = args.seed if args.seed[0] != 0 else [random.randint(1, 10000000) for _ in range(5)]
steps = args.steps
tags = args.tags
negative_time = args.negative_time



if args.wandb:
    wandb.login()
    wandb.config = {"prompt": prompt,
                    "negative prompt": negative_prompt, 
                    "seeds": seeds,
                    "steps": steps,
                    "tags": tags
                    }
    run = wandb.init(
        project=args.project,
        notes=args.note,
        group=args.group,
        config=wandb.config,
    )


for seed in iter(seeds):
    with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
        plt, fig, axs = get_plt(len(negative_time))
        for negative_time_index, negative_time_value in enumerate(negative_time):
            out = pipe(prompt, negative_prompt=negative_prompt if len(negative_prompt)> 0 else None, num_inference_steps=steps, generator=set_seed(seed), negative_time=negative_time_value)
            axs[math.floor(negative_time_index/4)][negative_time_index%4].imshow(out.images[0])
            axs[math.floor(negative_time_index/4)][negative_time_index%4].set_title(f"negative time: {negative_time_value}")
            
            # print(negative_time_value)
        if args.wandb:
            wandb.log({"pic": fig})
        else:
            plt.savefig('pic.png')
            