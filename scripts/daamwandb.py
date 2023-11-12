import argparse
from daam import trace, set_seed
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
import os
import wandb
import math
import random

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
args = parser.parse_args()

#os.environ["WANDB_MODE"] = "offline"

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

prompt = args.prompt
negative_prompt = args.negative_prompt
seeds = args.seed if args.seed[0] != 0 else [random.randint(1, 10000000) for _ in range(5)]
steps = args.steps

words = args.words.split(', ')
tags = args.tags
layer_id = args.layer_id
factors = args.factors
time_id = args.time_id
head_id = args.head_id



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
                    "tags": tags
                    }
    run = wandb.init(
        project=args.project,
        notes=args.note,
        group=args.group,
        config=wandb.config,
    )

layer_id = None if layer_id is None else list(range(layer_id[0], layer_id[1] + 1))
time_id = None if time_id is None else list(range(time_id[0], time_id[1] + 1))


for seed in iter(seeds):
    with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
        with trace(pipe) as tc:

            if len(negative_prompt)> 0:
                out = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=steps, generator=set_seed(seed))
            else:
                out = pipe(prompt, num_inference_steps=steps, generator=set_seed(seed))
            
            heat_map = tc.compute_global_heat_map(factors=factors, head_idx=head_id, layer_idx=layer_id, time_idx=time_id)
            #print(heat_map.negative_heat_maps.shape)
            heat_map = tc.compute_global_heat_map(factors=factors, head_idx=head_id, layer_idx=layer_id, time_idx=time_id)
            #print(heat_map.negative_heat_maps.shape)

            plt.clf()
            plt.rcParams.update({'font.size': 8})
            if len(words) < 4:
                fig, axs = plt.subplots(1, len(words) + 1, figsize=(5*(len(words)+1), 5+1))
                axs[0].imshow(out.images[0])
                for ax in axs:
                    ax.axis('off')
            else:
                fig, axs = plt.subplots(math.ceil((len(words) + 1) / 4),4)
                axs[0][0].imshow(out.images[0])

                for i in range(math.ceil((len(words) + 1) / 4)):
                    for j in range(4):
                        #print(i,j)
                        ax = axs[i][j]
                        #print(ax)
                        ax.axis('off')
                
            
            # text = '' if layer_id is None else 'layer_id: ' + str(layer_id[0]) + ' ' + str(layer_id[-1])
            # text += '' if factors is None else ' factors: ' + str(factors)
            # text += '' if time_id is None else ' time_id: ' + str(time_id[0]) + ' ' + str(time_id[-1])
            # text += '' if head_id is None else ' head_id: ' + str(head_id)
            
            # axs[0].set_title(text)  # 调整这里的y值和字体

            for i, word in enumerate(words):
                heat_map_word = heat_map.compute_word_heat_map(word)
                if len(words) < 4:
                    heat_map_word.plot_overlay(out.images[0], ax=axs[i+1])
                else:
                    #print(math.floor((i+1)/4),(i+1)%4)
                    heat_map_word.plot_overlay(out.images[0], ax=axs[math.floor((i+1)/4)][(i+1)%4])


            plt.subplots_adjust(top=0.95)
            if args.wandb:
                wandb.log({"pic": fig})
            else:
                plt.savefig('pic.png')