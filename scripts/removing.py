import argparse
from diffusers import StableDiffusionPipeline
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
parser.add_argument('--seeds', type=int, nargs='+', default=None)
parser.add_argument('--steps', type=int, default=30)
parser.add_argument('--wandb',action='store_true',help='use wandb')
parser.add_argument('--estimated_time', type=int, nargs = '+', default=None)
parser.add_argument('--negative_time', type=int, nargs = '+', default=None)
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
estimated_time = args.estimated_time
negative_time = args.negative_time
seeds = args.seeds


wandb.login()
if args.wandb:
    wandb.login()
    wandb.config = {"prompt": prompt,
                    "negative prompt": negative_prompt, 
                    "seeds": seeds,
                    "steps": steps,
                    "negative time": negative_time,
                    "estimated time": estimated_time,
                    }
    run = wandb.init(
        project=args.project,
        notes=args.note,
        group=args.group,
        config=wandb.config,
    )

fig, axs = plt.subplots(1,3)
axs = axs.reshape(1,-1)
fig.subplots_adjust(wspace=.1, hspace=.2)
fig.set_size_inches(30,10)
for i,seed in enumerate(seeds):
    with torch.no_grad():
        out = pipe(prompt=prompt, generator=set_seed(seed), num_inference_steps=steps)
        axs[0][0].axis('off')
        axs[0][0].imshow(out.images[0])
        axs[0][0].set_title(f"{seed}",fontsize=20)
        
        out = pipe(prompt=prompt,negative_prompt = negative_prompt, generator=set_seed(seed), num_inference_steps=steps)
        axs[0][1].axis('off')
        axs[0][1].imshow(out.images[0])
        axs[0][1].set_title(f"{seed}",fontsize=20)
        
        out = pipe(prompt=prompt,negative_prompt = negative_prompt, generator=set_seed(seed), num_inference_steps=steps, negative_time=negative_time)
        axs[0][2].axis('off')
        axs[0][2].imshow(out.images[0])
        axs[0][2].set_title(f"{negative_time}",fontsize=20)
            
    if args.wandb:
        wandb.log({"plt": fig}) 
    else:
        # save image fig
        fig.savefig(f"pics/plt.png")
            