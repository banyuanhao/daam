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
import matplotlib.patches as patches
import numpy as np

def vector_projection(a, b):
    """
    Project vector a onto vector b.
    """
    return (torch.dot(a, b) / torch.dot(b, b)) * b
    

def projection(x, y):
    batch_size, channels, width, height = x.shape
    x = x.view(channels, width * height)
    y = y.view(channels, width * height)
    
    proj = torch.zeros_like(x)
    
    if batch_size > 1:
        raise NotImplementedError("Batch size > 1 not implemented")
    for i in range(channels):
        proj[i,] = vector_projection(x[i,], y[i,])
        
    x = x.view(batch_size, channels, width, height)
    y = y.view(batch_size, channels, width, height)
    proj = proj.view(batch_size, channels, width, height)
    
    return proj

def projectionalliinone(x, y):
    batch_size, channels, width, height = x.shape
    x = x.reshape(channels * width * height)
    y = y.reshape(channels * width * height)
    
    proj = torch.zeros_like(x)
    
    if batch_size > 1:
        raise NotImplementedError("Batch size > 1 not implemented")
    proj = vector_projection(x, y)
        
    x = x.view(batch_size, channels, width, height)
    y = y.view(batch_size, channels, width, height)
    proj = proj.view(batch_size, channels, width, height)
    
    return proj
    
    
    
    
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
parser.add_argument('--negative_time', type=int, required=True)

parser.add_argument('--project', type=str, default='negative prompt')
parser.add_argument('--note', type=str, default='negative prompt')
parser.add_argument('--group', type=str, required=True)
parser.add_argument('--tags', metavar='S', type=str, nargs='+',default='negative prompt')
parser.add_argument('--steps', type=int, default=30)
parser.add_argument('--wandb',action='store_true',help='use wandb')
parser.add_argument('--bound_box',type = int, nargs='+', required=True)
args = parser.parse_args()

#os.environ["WANDB_MODE"] = "offline"
wandb.login()

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipelineForNegativePrompts.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)
bound_box = args.bound_box

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
                    "tags": tags,
                    "negative time": negative_time,
                    "bound box": bound_box,
                    }
    run = wandb.init(
        project=args.project,
        notes=args.note,
        group=args.group,
        config=wandb.config,
    )


save_dict = {}



for i,seed in enumerate(iter(seeds)):
    
    with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
        
        
        ### plot negative time plot
        fig, axs = plt.subplots(3, 3, figsize=(10, 5+1))
        
        out, _, _, _, _ = pipe.negative_accumulate(prompt, negative_prompt=negative_prompt if len(negative_prompt)> 0 else None, num_inference_steps=steps, generator=set_seed(seed),negative_time=negative_time,output_type='image')
        
        axs[0][0].imshow(out.images[0])
        
        bound_box_image = [tmp*8 for tmp in bound_box]
        mask_image = np.zeros_like(out.images[0])
        mask_image[bound_box_image[1]:bound_box_image[1]+bound_box_image[3],bound_box_image[0]:bound_box_image[0]+bound_box_image[2],:] = 1
        axs[0][2].imshow((out.images[0]*mask_image))
        
        
        ### compute with negative prompt
        out, diffusion_process, negative_noises, positive_noises, uncond_noises = pipe.negative_accumulate(prompt, negative_prompt=negative_prompt if len(negative_prompt)> 0 else None, num_inference_steps=steps, generator=set_seed(seed),negative_time=40,output_type='image')
        
        axs[1][0].imshow(out.images[0])
        
        diffusion_process_draw = []
        for k in range(len(diffusion_process)):
            temp = torch.mean(diffusion_process[k],dim=1,keepdim=True)
            temp = torch.cat([temp]*3,dim=1)
            # scale temp to 0-1
            # temp = (temp - torch.min(temp))/(torch.max(temp) - torch.min(temp))
            diffusion_process_draw.append(temp)
                    
        ## bounding box and mask
        mask_latent = torch.zeros_like(diffusion_process[0]).to(device)
        mask_latent[:, :,bound_box[1]:bound_box[1]+bound_box[3],bound_box[0]:bound_box[0]+bound_box[2]] = 1
        

        axs[1][1].imshow((diffusion_process_draw[30]*mask_latent[0,0:3]).squeeze(0).cpu().numpy().transpose(1, 2, 0))
        
        mask_image[bound_box_image[1]:bound_box_image[1]+bound_box_image[3],bound_box_image[0]:bound_box_image[0]+bound_box_image[2],:] = 1
        axs[1][2].imshow((out.images[0]*mask_image))
        
        ratiowith = [float(torch.norm(projectionalliinone(positive_noises[k][:, :,bound_box[1]:bound_box[1]+bound_box[3],bound_box[0]:bound_box[0]+bound_box[2]],negative_noises[k][:, :,bound_box[1]:bound_box[1]+bound_box[3],bound_box[0]:bound_box[0]+bound_box[2]]))) for k in range(len(positive_noises))]
        
        ### compute without negative prompt
        out, diffusion_process_, negative_noises_, positive_noises_, uncond_noises_ = pipe.negative_accumulate(prompt, negative_prompt=negative_prompt if len(negative_prompt)> 0 else None, num_inference_steps=steps, generator=set_seed(seed),negative_time=0,output_type='image')
        
        axs[2][0].imshow(out.images[0])
        
        diffusion_process_draw_ = []
        for k in range(len(diffusion_process_)):
            temp = torch.mean(diffusion_process_[k],dim=1,keepdim=True)
            temp = torch.cat([temp]*3,dim=1)
            # scale temp to 0-1
            # temp = (temp - torch.min(temp))/(torch.max(temp) - torch.min(temp))
            diffusion_process_draw_.append(temp)
                    
        ## bounding box and mask
        mask_latent_ = torch.zeros_like(diffusion_process_[0]).to(device)
        mask_latent_[:, :,bound_box[1]:bound_box[1]+bound_box[3],bound_box[0]:bound_box[0]+bound_box[2]] = 1
        

        axs[2][1].imshow((diffusion_process_draw_[30]*mask_latent_[0,0:3]).squeeze(0).cpu().numpy().transpose(1, 2, 0))
        
        bound_box_image_= [tmp*8 for tmp in bound_box]
        mask_image_ = np.zeros_like(out.images[0])
        mask_image_[bound_box_image[1]:bound_box_image[1]+bound_box_image[3],bound_box_image[0]:bound_box_image[0]+bound_box_image[2],:] = 1
        axs[2][2].imshow((out.images[0]*mask_image))
          
        # rect = patches.Rectangle((bound_box[0], bound_box[1]), bound_box[2], bound_box[3], linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)
        ratiowithout = [float(torch.norm(projectionalliinone(positive_noises_[k][:, :,bound_box[1]:bound_box[1]+bound_box[3],bound_box[0]:bound_box[0]+bound_box[2]],negative_noises_[k][:, :,bound_box[1]:bound_box[1]+bound_box[3],bound_box[0]:bound_box[0]+bound_box[2]]))) for k in range(len(positive_noises))]
        
        diff = [ratiowith[k] - ratiowithout[k] for k in range(len(ratiowithout))]
    
        axs[0][1].plot(diff[0:10])
        print(diff)
            
                    
        if args.wandb:
            wandb.log({f"ratio: {str(seed)}": fig})
        else:
            plt.savefig('pics/pic.png')