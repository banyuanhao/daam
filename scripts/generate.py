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


# pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)
# print(pipe.scheduler)

# pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(device)    

# pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")

# refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(device)


prompt = args.prompt
#prompt = ["glasse"]
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



# with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
#     out = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps, generator=set_seed(seed))
if len(negative_time) == 1:
    fig, axs = plt.subplots(len(negative_time),len(estimated_time)) 
    axs = axs.reshape(1,-1)
else:
    fig, axs = plt.subplots(len(estimated_time),len(negative_time)) 
# set the spacing between axes.
fig.subplots_adjust(wspace=.1, hspace=.2)
fig.set_size_inches(len(estimated_time)*10, len(negative_time)*10)
for seed in seeds:
    for i, estimated_t in enumerate(estimated_time):
        for j, negative_t in enumerate(negative_time):
            with torch.no_grad():
                if estimated_t >= negative_t and len(negative_time) == 1:
                    continue
                out = pipe.one_step_estimation(prompt=prompt,negative_prompt = negative_prompt, generator=set_seed(seed), num_inference_steps=steps,estimate_time=estimated_t,negative_time=negative_t)
                if len(negative_time) == 1:
                    axs[j][i].axis('off')
                    axs[j][i].imshow(out.images[0])
                    axs[j][i].set_title(f"{estimated_t} {negative_t}",fontsize=20)
                else:      
                    axs[i][j].axis('off')
                    axs[i][j].imshow(out.images[0])
                    axs[i][j].set_title(f"{estimated_t} {negative_t}",fontsize=20)
                    
            
    if args.wandb:
        wandb.log({"plt": fig}) 
    else:
        # save image fig
        fig.savefig(f"plt.png")