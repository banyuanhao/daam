# file to generate images from prompts and seeds

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image,  StableDiffusionXLImg2ImgPipeline
import torch
import random
import numpy as np
from typing import TypeVar
T = TypeVar('T')
from PIL import Image
from pathlib import Path
import os
from tqdm import tqdm
from matplotlib import pyplot as plt

def auto_device(obj: T = torch.device('cpu')) -> T:
    if isinstance(obj, torch.device):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        return obj.to('cuda')

def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    gen = torch.Generator(device=auto_device())
    gen.manual_seed(seed)

    return gen
# seed = 17382
# class_names = 'sports_ball.txt'
# print(f'classes: {class_names[:-4]}')
# seed = 32881
# class_names = 'stop_sign.txt'
seed = 53725
class_names = 'handbag.txt'


model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)  

dataset_path = Path(f'dataset/ODFN/version_2')
prompts_path = dataset_path/'prompts_10000_5_5'
prompt_path = prompts_path/class_names
latents = pipe.get_latents(prompt='chuchu', generator=set_seed(seed))

with torch.no_grad():
    fig, axs = plt.subplots(2, 5, figsize=(5*5, 5*2))
    with open(prompt_path, 'r') as f:
        prompts = f.read()
        prompts = prompts.split('\n')
        name = prompts[0]
        prompts = prompts[1:]
    for k, prompt in enumerate(prompts):
        out = pipe(prompt=prompt, generator=set_seed(seed),latents = latents)
        axs[0][k].imshow(out.images[0])
    # flip the latent
    latents = latents.flip(3)
    for k, prompt in enumerate(prompts):
        out = pipe(prompt=prompt, generator=set_seed(seed),latents = latents)
        axs[1][k].imshow(out.images[0])
    fig.savefig(f'pics/flip_{class_names[:-5]}_{seed}.png')