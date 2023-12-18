# a script to test the changing of the negative prompt
import argparse
from daam import set_seed, trace
from models.diffuserpipeline import StableDiffusionPipelineForNegativePrompts
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
import os
import wandb
import math
import random
import torch.nn.functional as F

prompt_sec = "train station, watercolor painting" 
negative_prompt_sec = "train"

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
    x = x.view(channels * width * height)
    y = y.view(channels * width * height)
    
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
pipe_ori = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)
pipe_ori = pipe_ori.to(device)

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

save_dict = {}
placehold = torch.zeros(len(seeds), len(negative_time))
for i,seed in enumerate(iter(seeds)):
    
    with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():

        plt, fig, axs = get_plt(2)
            
        out, diffusion_process, negative_noises, positive_noises, uncond_noises = pipe.negative_accumulate(prompt, negative_prompt=negative_prompt if len(negative_prompt)> 0 else None, num_inference_steps=steps, generator=set_seed(seed),negative_time=40)
        
        with trace(pipe_ori) as tc:
            out_ori = pipe_ori(prompt, negative_prompt=negative_prompt if len(negative_prompt)> 0 else None, num_inference_steps=steps, generator=set_seed(seed))
            negative_maps = tc.return_heat_map()
            # negative_maps = [torch.cat([negative_map]*3) for negative_map in negative_maps]
            #negative_maps = [negative_maps[i].to(torch.float32).transpose(0,1).transpose(1,2).cpu().numpy() for i in range(len(negative_maps))]
        
        negative_maps = [torch.cat([negative_map]*4).unsqueeze(0) for negative_map in negative_maps]
    
        
        gro = [2]
        for i in range(31):
            positive_noise = positive_noises[i] # * negative_maps[i]
            
            negative_noise = negative_noises[i] # * negative_maps[i]
            
            # compute cos similarity
            positive_noise = positive_noise.view(4 * 64 * 64)
            negative_noise = negative_noise.view(4 * 64 * 64)
            cos_sim = F.cosine_similarity(positive_noise, negative_noise, dim=0)
            print(cos_sim)
            
            
            
        out, diffusion_process, negative_noises, positive_noises, uncond_noises = pipe.negative_accumulate(prompt, negative_prompt=negative_prompt if len(negative_prompt)> 0 else None, num_inference_steps=steps, generator=set_seed(seed),negative_time=0)
        
        fig, ax = plt.subplots()
        # turn off x and y axis
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(out.images[0])
        fig.savefig(f'out_{negative_prompt}.png')
        # raise NotImplementedError("Batch size > 1 not implemented")
        
        
        with trace(pipe_ori) as tc:
            out_ori = pipe_ori(prompt, negative_prompt=negative_prompt if len(negative_prompt)> 0 else None, num_inference_steps=steps, generator=set_seed(seed))
            negative_maps = tc.return_heat_map()
            # negative_maps = [torch.cat([negative_map]*3) for negative_map in negative_maps]
            #negative_maps = [negative_maps[i].to(torch.float32).transpose(0,1).transpose(1,2).cpu().numpy() for i in range(len(negative_maps))]
        
        negative_maps = [torch.cat([negative_map]*4).unsqueeze(0) for negative_map in negative_maps]
    
        
        gro = [2,3]
        for i in gro:
            positive_noise = positive_noises[i]  * negative_maps[i]
            
            negative_noise = negative_noises[i] * negative_maps[i]
            
            # compute cos similarity
            positive_noise = positive_noise.view(4 * 64 * 64)
            negative_noise = negative_noise.view(4 * 64 * 64)
            noise = torch.randn(4 * 64 * 64).to(device)
            # cos_sim = F.cosine_similarity(positive_noise, negative_noise, dim=0)
            cos_sim = F.cosine_similarity(positive_noise, noise, dim=0)
            print(cos_sim)
            
        
        # fig, ax = plt.subplots()
        # ax.imshow(negative_maps[3])
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # fig.savefig(f'negative_maps_{negative_prompt}.png')

          
        #ratio40 = [float(torch.norm(projectionalliinone(positive_noises[i],negative_noises[i]))) for i in range(len(positive_noises))]
        
        
            
        # out, diffusion_process, negative_noises, positive_noises, uncond_noises = pipe.negative_accumulate(prompt, negative_prompt=negative_prompt if len(negative_prompt)> 0 else None, num_inference_steps=steps, generator=set_seed(seed),negative_time=0)
            
        # ratio0 = [float(torch.norm(projectionalliinone(positive_noises[i],negative_noises[i]))) for i in range(len(positive_noises))]
            
#             diff = [ratio0[i] - ratio40[i] for i in range(len(ratio0))]
#             placehold[i] = torch.tensor(diff)

# save_dict["seed"] = seeds
# save_dict["placehold"] = placehold
# save_dict["negative_prompt"] = negative_prompt
# torch.save(save_dict, f'proj_{negative_prompt}.pt')


            