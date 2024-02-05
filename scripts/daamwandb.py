import argparse
from daam import trace, set_seed
from diffusers import StableDiffusionPipeline
from models.diffuserpipeline import StableDiffusionPipelineForNegativePrompts
import torch
import matplotlib.pyplot as plt
import os
import wandb
import math
import random

def get_plt(num):
    plt.clf()
    plt.rcParams.update({'font.size': 12})
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
parser.add_argument('--head_id', type=int, nargs='+', default=None)
parser.add_argument('--layer_id', type=int, nargs='+', default=None)
parser.add_argument('--time_id', type=int, nargs='+', default=None)
parser.add_argument('--factors', type=int, nargs='+', default=None)

parser.add_argument('--project', type=str, default='negative prompt')
parser.add_argument('--note', type=str, default='negative prompt')
parser.add_argument('--group', type=str, required=True)
parser.add_argument('--words', type=str,required=True)
parser.add_argument('--tags', metavar='S', type=str, nargs='+',default='negative prompt')
parser.add_argument('--steps', type=int, default=30)
parser.add_argument('--wandb',action='store_true',help='use wandb')
parser.add_argument('--negative_time', type=int, nargs='+',  default=None)
args = parser.parse_args()

#os.environ["WANDB_MODE"] = "offline"
wandb.login()

prompt = args.prompt
negative_prompt = args.negative_prompt
seeds = args.seed if args.seed[0] != 0 else [random.randint(1, 10000000) for _ in range(5)]
steps = args.steps

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'

pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

words = args.words.split(', ')
tags = args.tags
layer_id = args.layer_id
factors = args.factors
time_id = args.time_id
head_id = args.head_id
negative_time = args.negative_time

if args.wandb:
    wandb.login()
    wandb.config = {"prompt": prompt,
                    "negative prompt": negative_prompt, 
                    "seeds": seeds,
                    "word": words,
                    "layer_id": layer_id,
                    "factors": factors,
                    "time_id": time_id,
                    "head_id": head_id,
                    "steps": steps,
                    "tags": tags,
                    "negative_time": negative_time
                    }
    run = wandb.init(
        project=args.project,
        notes=args.note,
        group=args.group,
        config=wandb.config,
    )

layer_id = None if layer_id is None else [layer_id[i:i + 2] for i in range(0, len(layer_id), 2)]
layer_id = None if layer_id is None else [list(range(layer_id_[0], layer_id_[1] + 1)) for layer_id_ in layer_id] 
time_id = None if time_id is None else [time_id[i:i + 2] for i in range(0, len(time_id), 2)]
time_id = None if time_id is None else [list(range(time_id_[0], time_id_[1] + 1)) for time_id_ in time_id]

for seed in iter(seeds):
    with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
        with trace(pipe) as tc:
            if len(negative_prompt)> 0:
                out = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=steps, generator=set_seed(seed), negative_time=negative_time)
            else:
                out = pipe(prompt, num_inference_steps=steps, generator=set_seed(seed))
                
            if layer_id is None and time_id is None:
                plt, fig, axs = get_plt(len(words)+1)
                fig.suptitle(f'{negative_time}')
                if len(words) < 4:
                    axs[0].imshow(out.images[0])
                else:
                    axs[0][0].imshow(out.images[0])
                heat_map = tc.compute_global_heat_map(factors=factors)
                for i, word in enumerate(words):
                    heat_map_word = heat_map.compute_word_heat_map(word)
                    if len(words) < 4:
                        heat_map_word.plot_overlay(out.images[0], ax=axs[i+1])
                    else:
                        heat_map_word.plot_overlay(out.images[0], ax=axs[math.floor((i+1)/4)][(i+1)%4])
                        
            elif layer_id is not None and time_id is None: 
                if len(words) > 1:
                        raise ValueError(f'Only one word is supported, but {len(words)} words are given!')
                plotnum = len(layer_id) + 1
                plt, fig, axs = get_plt(plotnum)
                if len(layer_id) < 4:
                    axs[0].imshow(out.images[0])
                    axs[0].set_title(words[0])
                else:
                    axs[0][0].imshow(out.images[0])
                    axs[0][0].set_title(words[0])
                    
                for i, layer_i in enumerate(layer_id):
                    fig.suptitle(f'{prompt} {negative_prompt}')
                    heat_map = tc.compute_global_heat_map(factors=factors, layer_idx=layer_i)
                    heat_map_word = heat_map.compute_word_heat_map(words[0])
                    if len(layer_id) < 4:
                        heat_map_word.plot_overlay(out.images[0], ax=axs[i+1])
                        axs[i+1].set_title('layer_id: ' + str(time_i[0]) + ' ' + str(time_i[-1]))
                    else:
                        heat_map_word.plot_overlay(out.images[0], ax=axs[math.floor((i+1)/4)][(i+1)%4])
                        axs[math.floor((i+1)/4)][(i+1)%4].set_title('layer_id: ' + str(layer_i[0]) + ' ' + str(layer_i[-1]))
                        
            elif layer_id is None and time_id is not None:
                if len(words) > 1:
                        raise ValueError(f'Only one word is supported, but {len(words)} words are given!')
                plotnum = len(time_id) + 1
                plt, fig, axs = get_plt(plotnum)
                if len(time_id) < 4:
                    axs[0].imshow(out.images[0])
                    axs[0].set_title(words[0])
                else:
                    axs[0][0].imshow(out.images[0])
                    axs[0][0].set_title(words[0])
                    
                for i, time_i in enumerate(time_id):
                    heat_map = tc.compute_global_heat_map(factors=factors, time_idx=time_i)
                    heat_map_word = heat_map.compute_word_heat_map(words[0])
                    if len(time_id) < 4:
                        heat_map_word.plot_overlay(out.images[0], ax=axs[i+1])
                        axs[i+1].set_title('time_id: ' + str(time_i[0]) + ' ' + str(time_i[-1]))
                    else:
                        heat_map_word.plot_overlay(out.images[0], ax=axs[math.floor((i+1)/4)][(i+1)%4])
                        axs[math.floor((i+1)/4)][(i+1)%4].set_title('time_id: ' + str(time_i[0]) + ' ' + str(time_i[-1]))
            else:
                raise ValueError(f'not supported!')
                
            if args.wandb:
                wandb.log({"pic": fig})
            else:
                plt.savefig('pics/pic.png')
                    