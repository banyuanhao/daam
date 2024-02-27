import argparse
from diffusers import StableDiffusionPipeline
# StableDiffusionXLPipeline, AutoPipelineForText2Image,  StableDiffusionXLImg2ImgPipeline
from models.diffuserpipeline import StableDiffusionPipelineForNegativePrompts
import torch
import random
import numpy as np
from typing import TypeVar
T = TypeVar('T')
from PIL import Image
import wandb
import matplotlib.pyplot as plt

def auto_device(obj: T = torch.device('cpu')) -> T:
    if isinstance(obj, torch.device):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        return obj.to('cuda')

    return obj

def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    gen = torch.Generator(device=auto_device())
    gen.manual_seed(seed)

    return gen
    

parser = argparse.ArgumentParser(description='Diffusion')
parser.add_argument('--prompt', type=str, required=True)
parser.add_argument('--negative_prompt', type=str,default=None)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--steps', type=int, default=30)
parser.add_argument('--wandb',action='store_true',help='use wandb')
parser.add_argument('--start_time', type=int, nargs = '+', default=None)
parser.add_argument('--end_time', type=int, nargs = '+', default=None)
parser.add_argument('--project', type=str, default='negative prompt')
parser.add_argument('--note', type=str, default='negative prompt')
parser.add_argument('--group', type=str, required=True)
args = parser.parse_args()

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'

pipe = StableDiffusionPipelineForNegativePrompts.from_pretrained(model_id, use_auth_token=True).to(device)


prompt = args.prompt
steps = args.steps
negative_prompt = args.negative_prompt
end_time = args.end_time
start_time = args.start_time
seed = args.seed


wandb.login()
if args.wandb:
    wandb.login()
    wandb.config = {"prompt": prompt,
                    "negative prompt": negative_prompt, 
                    "seed": seed,
                    "steps": steps,
                    "start time": start_time,
                    "end time": end_time,
                    }
    run = wandb.init(
        project=args.project,
        notes=args.note,
        group=args.group,
        config=wandb.config,
    )




fig, axs = plt.subplots(len(start_time),len(end_time)) 
fig.subplots_adjust(wspace=.1, hspace=.2)
fig.set_size_inches(len(start_time)*10, len(end_time)*10)

for i, start_t in enumerate(start_time):
    for j, end_t in enumerate(end_time):
        with torch.no_grad():
            if start_t >= end_t:
                axs[i][j].axis('off')
                continue
            negative_time = list(range(start_t,end_t))
            out = pipe(prompt=prompt,negative_prompt = negative_prompt, generator=set_seed(seed), num_inference_steps=steps, negative_time=negative_time)
            axs[i][j].axis('off')
            axs[i][j].imshow(out.images[0])
            axs[i][j].set_title(f"{start_t} {end_t}",fontsize=20)
            

out = pipe(prompt=prompt, generator=set_seed(seed), num_inference_steps=steps)
axs[len(start_time)-1][0].imshow(out.images[0])
        
if args.wandb:
    wandb.log({"plt": fig}) 
else:
    # save image fig
    fig.savefig(f"pics/plt.png")
    
    
# python scripts/gridremoving.py --seed 0 --prompt "professional office woman" --negative_prompt glasses --start_time 0 2 4 6 8 10 12 14 --end_time 1 3 5 7 9 11 13 15 31 --group wrapup_test