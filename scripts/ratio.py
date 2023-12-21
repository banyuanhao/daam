import argparse
from daam import trace, set_seed
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
import os
import wandb
import math
import random
import numpy as np
import cv2
    
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
parser.add_argument('--bound_box',type = int, nargs='+', default=None)

args = parser.parse_args()

#os.environ["WANDB_MODE"] = "offline"
wandb.login()

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

prompt = args.prompt
negative_prompt = args.negative_prompt
seeds = args.seed if args.seed[0] != 0 else [random.randint(1, 10000000) for _ in range(5)]
steps = args.steps
tags = args.tags
bound_box = args.bound_box



if args.wandb:
    wandb.login()
    wandb.config = {"prompt": prompt,
                    "negative prompt": negative_prompt, 
                    "seeds": seeds,
                    "steps": steps,
                    "tags": tags,
                    "bound box": bound_box,
                    }
    run = wandb.init(
        project=args.project,
        notes=args.note,
        group=args.group,
        config=wandb.config,
    )

bound_box_image = [tmp*8 for tmp in bound_box]


for seed in iter(seeds):
    with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
        with trace(pipe) as tc:
            
            fig, axs = plt.subplots(2, 2, figsize=(10, 10+1))
            
            out = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=steps, generator=set_seed(seed),output_type='latent')
            
            image = pipe.decode_latents(out.images)[0]
            latent = out.images
            
            latent = latent.mean(dim=1,keepdim=True)
            latent = torch.cat([latent]*3,dim=1)[0]
            
            mask_latent = torch.zeros_like(latent).to(device)
            mask_latent[:,bound_box[1]:bound_box[1]+bound_box[3],bound_box[0]:bound_box[0]+bound_box[2]] = 1
            
            
            mask_image = np.zeros_like(image)
            mask_image[bound_box_image[1]:bound_box_image[1]+bound_box_image[3],bound_box_image[0]:bound_box_image[0]+bound_box_image[2],:] = 1
            
            
            axs[0][0].imshow(image)
            axs[0][0].set_title('Image')
            
            axs[0][1].imshow((image*mask_image))
            axs[0][1].set_title('Image')
            
            pos, neg = tc.compute_activation_ratio()
            ratio = [neg[i]/pos[i] for i in range(len(pos))]
            axs[1][0].plot(ratio, label='ratio')
            axs[1][0].set_title(f'ratio')
            
            pos, neg = tc.compute_activation_ratio(bounding_box=bound_box)
            ratio = [neg[i]/pos[i] for i in range(len(pos))]
            axs[1][1].plot(ratio, label='ratio')
            axs[1][1].set_title(f'{bound_box}')
            
            if args.wandb:
                wandb.log({"Ratio": fig}) 
            else:
                fig.savefig('pics/ratio.png')
