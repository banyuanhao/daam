# a script to test the changing of the negative prompt
import argparse
from daam import set_seed
from models.diffuserpipeline import StableDiffusionPipelineForNegativePrompts, NegativeMapOutput
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

def get_axs(axs, id, num):
    if num < 4:
        return axs[id]
    else:
        return axs[math.floor(id/4)][id%4]

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
parser.add_argument('--look_time', type=int, nargs='+', default=0)
args = parser.parse_args()

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
look_time = args.look_time



if args.wandb:
    wandb.login()
    wandb.config = {"prompt": prompt,
                    "negative prompt": negative_prompt, 
                    "seeds": seeds,
                    "steps": steps,
                    "tags": tags,
                    "negative time": args.negative_time,
                    "look time": args.look_time
                    }
    run = wandb.init(
        project=args.project,
        notes=args.note,
        group=args.group,
        config=wandb.config,
    )
    
negative_time = [False] * 31
if args.negative_time is not None:
    for i in args.negative_time:
        negative_time[i] = True

for seed in iter(seeds):
    with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
        
        plt, fig, axs = get_plt(8)
        
        out = pipe(prompt, negative_prompt=negative_prompt if len(negative_prompt)> 0 else None, num_inference_steps=steps, generator=set_seed(seed))
        ax = get_axs(axs, 0, 8)
        ax.imshow(out.images[0])
        ax.set_title('without negative prompt')
        
        out = pipe(prompt, num_inference_steps=steps, generator=set_seed(seed))
        ax = get_axs(axs, 1, 8)
        ax.imshow(out.images[0])
        ax.set_title('with negative prompt')
        
        for i, time in enumerate(look_time):

            out = pipe.diff_map(prompt, negative_prompt=negative_prompt if len(negative_prompt)> 0 else None, num_inference_steps=steps, generator=set_seed(seed), negative_time=negative_time, look_step=time)
            
            
            difference = (out.image_withneg_2 - out.image_withoutneg_2) * 10 + 0.5
            ax = get_axs(axs, 2+i, 8) 
            ax.imshow(pipe.numpy_to_pil(difference)[0])
            ax.set_title(f'{time}')
        
        
        if args.wandb:
            wandb.log({"pic": fig})
        else:
            plt.savefig('pic.png')
            
            
            
# self.image_original = image_original
# self.image_withoutneg_1 = image_withoutneg_1
# self.image_withneg_1 = image_withneg_1
# self.image_withoutneg_2 = image_withoutneg_2
# self.image_withneg_2 = image_withneg_2